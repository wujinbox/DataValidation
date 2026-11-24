"""Loopback-only web control plane. Worker credentials stay on the cluster."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
import json
import math
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from catalog import DATA, ROOT, read, write
from cluster import (
    DATA_PYTHON, HOSTS, PYTHON, PUBLISH, REMOTE, inventory, probe, remote_run, ssh,
)
from control import stop_tree
from progress import progress
from datapipe import store, qc, queue
from datapipe.recipe import validate as validate_recipe

PORT = int(os.environ.get("WAN_STUDIO_PORT", "28765"))
RUNS = ROOT / "runs"
CATEGORIES = {c["id"]: c for c in store.read(DATA / "catalog.json", [])}
LOCK = threading.Lock()
RESOURCE_LOCK = threading.Lock()
# A page token survives host/origin rewriting without trusting forwarded headers.
PAGE_CSRF_TOKEN = secrets.token_urlsafe(32)
resource_cache = (0, [])


def resources():
    global resource_cache
    with RESOURCE_LOCK:
        if time.monotonic() - resource_cache[0] > 15 or not resource_cache[1]:
            resource_cache = (time.monotonic(), inventory())
        return resource_cache[1]


def logtail(path, size=5000):
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - size))
        return f.read().decode(errors="replace")


def status(run):
    cfg = read(run / "config.json")
    if (run / "execution.json").exists():
        cfg.update(read(run / "execution.json"))
    state = read(run / "state.json")
    if (run / "cancel.request").exists() and state["stage"] != "已停止":
        state = dict(stage="正在停止")
        if (run / "cancel-error.json").exists():
            state = dict(
                stage="停止待重试", error=read(run / "cancel-error.json")["error"]
            )
    metrics = progress(run, cfg)
    return dict(
        id=run.name,
        progress=metrics,
        can_stop=state["stage"] not in ("已停止", "完成", "失败", "正在停止"),
        config=cfg,
        **state,
        split=read(run / "split.json") if (run / "split.json").exists() else None,
        videos=metrics["pairs"],
        tail=logtail(run / "job.log"),
        can_publish=PUBLISH and (run / "evaluated.json").exists() and state["stage"] == "失败",
    )


def history():
    return [
        status(p)
        for p in sorted(RUNS.iterdir(), reverse=True)
        if (p / "state.json").exists()
    ]


def launch(run):
    with (run / "job.log").open("a") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "job.py"), str(run)],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    write(run / "process.json", dict(pid=proc.pid))


def integer(body, key, low, high):
    v = body.get(key)
    if type(v) is not int or not low <= v <= high:
        raise ValueError(f"{key} 超出范围 {low}–{high}")
    return v


def submit(body):
    category = CATEGORIES.get(body.get("category"))
    host = body.get("host")
    cards = body.get("gpus")
    if not category or not category["actions"]:
        raise ValueError("该 L2 缺少可核对的产品动作定义")
    if (
        host not in HOSTS
        or not isinstance(cards, list)
        or not cards
        or len(cards) > 8
        or any(type(i) is not int or not 0 <= i < 8 for i in cards)
        or len(set(cards)) != len(cards)
    ):
        raise ValueError("请选择同一节点上的空闲 GPU")
    lr = body.get("lr")
    if type(lr) not in (float, int) or not math.isfinite(lr) or not 0 < lr <= 0.001:
        raise ValueError("学习率须在 (0, 0.001] 内")
    cfg = dict(
        category=category["id"],
        name=category["name"],
        host=host,
        gpus=sorted(cards),
        lr=lr,
        batch=integer(body, "batch", 1, 6),
        products=integer(body, "products", 1, len(category["actions"])),
        steps=integer(body, "steps", 0, 100000),
        limit=integer(body, "limit", 0, 500000),
        sampling_steps=integer(body, "sampling_steps", 2, 100),
        seed=integer(body, "seed", 0, 2147483647),
    )
    if cfg["limit"] and cfg["limit"] < len(cards) * cfg["batch"]:
        raise ValueError("数据上限不足一个全局 batch")
    with LOCK:
        if queue.conflicts(host, cards):
            raise ValueError("所选卡已被配方队列预留")
        current = probe(host)
        if not set(cards) <= {g["index"] for g in current["cards"] if g["idle"]}:
            raise ValueError("卡状态已变化，请刷新后选择")
        for job in history():
            if (
                job["stage"] not in ("完成", "失败", "已停止", "飞书归档")
                and host == job["config"]["host"]
                and set(cards) & set(job["config"]["gpus"])
            ):
                raise ValueError("所选卡已有本系统任务")
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        run = RUNS / (stamp + "-" + secrets.token_hex(3))
        run.mkdir()
        write(run / "config.json", cfg)
        write(run / "state.json", dict(stage="启动中"))
        try:
            launch(run)
        except Exception as e:
            write(run / "state.json", dict(stage="失败", error=str(e)))
            raise
    return status(run)


def heavy(command, batch, body, timeout=300):
    try:
        result = subprocess.run(
            [DATA_PYTHON, "-m", "datapipe.cli", command, batch.name],
            cwd=ROOT,
            input=json.dumps(body, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ValueError("处理超时，请刷新批次状态后重试") from None
    if result.returncode:
        try:
            error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        except (ValueError, KeyError, IndexError):
            error = "数据处理失败，请查看批次日志"
        raise ValueError(error)
    return json.loads(result.stdout)


def submit_prepared(batch, body):
    recipe, _ = validate_recipe(body.get("recipe"), store.read(batch / "stats.json"))
    if recipe["dataset"] != batch.name:
        raise ValueError("配方 dataset 与批次不一致")
    parse_id = body.get("parse_id")
    if parse_id and (
        not isinstance(parse_id, str)
        or not store.ID.fullmatch(parse_id)
        or parse_id in (".", "..")
    ):
        raise ValueError("无效的解析记录 ID")
    host, cards = recipe["train"]["host"], recipe["train"]["gpus"]
    with LOCK:
        if queue.conflicts(host, cards):
            raise ValueError("所选 GPU 已被配方队列预留")
        current = probe(host)
        if not set(cards) <= {g["index"] for g in current["cards"] if g["idle"]}:
            raise ValueError("卡状态已变化，请刷新后选择")
        for job in history():
            if (
                job["stage"] not in ("完成", "失败", "已停止", "飞书归档")
                and host == job["config"]["host"]
                and set(cards) & set(job["config"]["gpus"])
            ):
                raise ValueError("所选卡已有本系统任务")
        result = heavy(
            "build", batch, dict(recipe=recipe, parse_id=parse_id, dispatch=True)
        )
        queue.enqueue(result["recipe_id"], result["runs"])
        return result


def stop(run):
    if queue.cancel_waiting(run):
        return status(run)
    current = status(run)
    if current["stage"] == "已停止":
        return current
    if not current["can_stop"] and current["stage"] != "正在停止":
        raise ValueError("实验已经结束")
    error_file = run / "cancel-error.json"
    error_file.unlink(missing_ok=True)
    (run / "cancel.request").touch()
    try:
        if not (run / "evaluated.json").exists():
            cmd = ssh(
                current["config"]["host"],
                [PYTHON, REMOTE + "/control.py", remote_run(run.name, current["config"])],
            )
            remote = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if remote.returncode:
                raise RuntimeError(remote.stderr[-600:] or "计算节点未确认停止")
        stop_tree(ROOT / "job.py", run)
        write(run / "state.json", dict(stage="已停止"))
    except Exception as exc:
        write(error_file, dict(error=str(exc)))
        raise
    return status(run)


class Handler(BaseHTTPRequestHandler):
    def send(self, code, data, mime="application/json"):
        raw = (
            json.dumps(data, ensure_ascii=False).encode()
            if mime == "application/json"
            else data
        )
        self.send_response(code)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def local(self):
        return self.headers.get("Host", "").split(":")[0] in ("localhost", "127.0.0.1")

    def stream_file(self, path, mime, ranges=False):
        if not path.is_file():
            return self.send(404, {"error": "文件不存在"})
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            start, end, code = 0, size - 1, 200
            requested = self.headers.get("Range") if ranges else None
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                try:
                    if not match or not any(match.groups()) or size == 0:
                        raise ValueError()
                    left, right = match.groups()
                    if left:
                        start = int(left)
                        end = min(int(right), size - 1) if right else size - 1
                        if start >= size or start > end:
                            raise ValueError()
                    else:
                        length = int(right)
                        if length <= 0:
                            raise ValueError()
                        start = max(0, size - length)
                    code = 206
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            stream.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    block = stream.read(min(65536, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        if not self.local():
            return self.send(403, {"error": "仅支持本地或 SSH 转发访问"})
        path = urlsplit(self.path).path
        try:
            if path == "/":
                return self.send(
                    200,
                    (ROOT / "index.html").read_bytes().replace(
                        b"__WAN_STUDIO_CSRF__", PAGE_CSRF_TOKEN.encode("ascii")
                    ),
                    "text/html",
                )
            if path == "/datapipe.js":
                return self.send(
                    200, (ROOT / "datapipe.js").read_bytes(), "text/javascript"
                )
            if path == "/api/categories":
                return self.send(200, list(CATEGORIES.values()))
            if path == "/api/resources":
                return self.send(200, resources())
            if path == "/api/jobs":
                return self.send(200, history())
            if path == "/api/datasets":
                return self.send(200, store.datasets())
            file_match = re.fullmatch(
                r"/api/jobs/(\d{8}-\d{6}-[0-9a-f]{6})/file/(mix.json|test_prompts.jsonl|cases.json|split.json|recipe.json)",
                path,
            )
            if file_match:
                mime = (
                    "application/x-ndjson"
                    if file_match[2].endswith("jsonl")
                    else "application/json"
                )
                return self.stream_file(RUNS / file_match[1] / file_match[2], mime)
            m = re.fullmatch(r"/api/datasets/([A-Za-z0-9_.-]{1,120})(?:/(.*))?", path)
            if m:
                batch = store.batch_path(m[1])
                suffix = m[2]
                if not suffix:
                    if store.read(batch / "state.json", {}).get("stage") in (
                        "登记",
                        "统计中",
                    ):
                        return self.send(200, store.detail(batch))
                    with store.lock(batch):
                        return self.send(200, store.detail(batch))
                if suffix == "report":
                    return self.stream_file(
                        batch / "分布报告.md", "text/markdown; charset=utf-8"
                    )
                chart = re.fullmatch(r"charts/([a-z0-9_]+\.(png|svg))", suffix)
                if chart:
                    return self.stream_file(
                        batch / "charts" / chart[1],
                        "image/png" if chart[2] == "png" else "image/svg+xml",
                    )
                video = re.fullmatch(r"video/(\d{1,6})", suffix)
                if video:
                    sample = store.read(batch / "qc/sample.json", {}).get("items", [])
                    idx = int(video[1])
                    if (
                        idx >= len(sample)
                        or sample[idx]["idx"] != idx
                        or not sample[idx].get("video_path")
                    ):
                        return self.send(404, {"error": "样本不存在"})
                    from pathlib import Path

                    return self.stream_file(
                        Path(sample[idx]["video_path"]), "video/mp4", True
                    )
            return self.send(404, {"error": "不存在"})
        except (FileNotFoundError, ValueError):
            return self.send(404, {"error": "不存在"})
        except Exception as exc:
            return self.send(500, {"error": str(exc)})

    def dataset_post(self, path, body):
        if path == "/api/datasets":
            n = body.get("n", 20)
            seed = body.get("seed", 42)
            if (
                type(n) is not int
                or not 1 <= n <= 1000
                or type(seed) is not int
                or not 0 <= seed <= 2147483647
            ):
                raise ValueError("无效的抽样数量或 seed")
            batch, created = store.register(body.get("path"))
            with store.lock(batch):
                state = store.read(batch / "state.json")
                if created or state["stage"] == "失败":
                    store.state(batch, "统计中", progress=0)
                    try:
                        with (batch / "ingest.log").open("a") as log:
                            subprocess.Popen(
                                [
                                    DATA_PYTHON,
                                    "-m",
                                    "datapipe.cli",
                                    "ingest",
                                    batch.name,
                                    "--n",
                                    str(n),
                                    "--seed",
                                    str(seed),
                                ],
                                cwd=ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True,
                            )
                    except Exception as exc:
                        store.state(batch, "失败", error=str(exc))
                        raise
            return self.send(
                202,
                dict(
                    batch_id=batch.name, stage=store.read(batch / "state.json")["stage"]
                ),
            )
        m = re.fullmatch(
            r"/api/datasets/([A-Za-z0-9_.-]{1,120})/(reviews|sample|qc/finish|recipe/parse|recipe/preview|recipe/submit)",
            path,
        )
        if not m:
            return self.send(404, {"error": "不存在"})
        batch = store.batch_path(m[1])
        action = m[2]
        if action == "reviews":
            return self.send(200, qc.save(batch, body))
        if action == "sample":
            extra = body.get("extra_l2")
            if not isinstance(extra, list) or not extra:
                raise ValueError("请选择追加抽检的 L2")
            # Sampling a full pool also runs outside the server process.
            result = heavy("sample-extra", batch, body)
            return self.send(200, result)
        if action == "qc/finish":
            return self.send(200, qc.finish(batch, body.get("override_reason")))
        if action == "recipe/parse":
            return self.send(200, heavy("recipe-parse", batch, body, 60))
        if action == "recipe/preview":
            return self.send(200, heavy("plan", batch, body))
        if action == "recipe/submit":
            return self.send(201, submit_prepared(batch, body))

    def do_POST(self):
        if (
            not self.local()
            or self.headers.get("X-Wan-Studio") != "1"
            or self.headers.get_content_type() != "application/json"
        ):
            return self.send(403, {"error": "须从本机工作台提交"})
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            # Only a page served by this process knows the unpredictable token.
            # Cross-origin pages cannot read it: no CORS permissions are granted.
            token = self.headers.get("X-Wan-Studio-Token", "")
            if not secrets.compare_digest(token.encode("utf-8"), PAGE_CSRF_TOKEN.encode("ascii")):
                return self.send(403, {"error": "页面校验已失效，请刷新工作台后重新提交（尚未保存）"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            path = urlsplit(self.path).path
            limit = (
                65536
                if path == "/api/datasets" or path.startswith("/api/datasets/")
                else 4096
            )
            if not 0 < n <= limit:
                raise ValueError("无效的请求长度")
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError("无效请求")
            if path == "/api/datasets" or path.startswith("/api/datasets/"):
                return self.dataset_post(path, body)
            if self.path == "/api/jobs":
                return self.send(201, submit(body))
            stop_match = re.fullmatch(
                r"/api/jobs/(\d{8}-\d{6}-[0-9a-f]{6})/stop", self.path
            )
            if stop_match:
                with LOCK:
                    result = stop(RUNS / stop_match[1])
                return self.send(200, result)
            m = re.fullmatch(r"/api/jobs/(\d{8}-\d{6}-[0-9a-f]{6})/publish", self.path)
            if m:
                run = RUNS / m[1]
                with LOCK:
                    if not status(run)["can_publish"]:
                        raise ValueError("此实验当前不可重试发布")
                    write(run / "state.json", dict(stage="飞书归档"))
                    launch(run)
                return self.send(202, status(run))
            self.send(404, {"error": "不存在"})
        except (ValueError, TypeError, KeyError, FileNotFoundError) as e:
            self.send(400, {"error": str(e)})
        except Exception as e:
            self.send(500, {"error": str(e)})


if __name__ == "__main__":
    RUNS.mkdir(exist_ok=True)
    print(f"Wan L2 Studio http://127.0.0.1:{PORT}", flush=True)
    http = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    queue.start(launch, LOCK)
    http.serve_forever()
