"""Persistent sequential recipes; reconciliation never blindly relaunches a job."""

from pathlib import Path
import threading
import time
from catalog import ROOT
from . import store

RUNS = ROOT / "runs"
QUEUE = ROOT / "queue.json"
TERMINAL = {"完成", "失败", "已停止"}


def entries():
    return store.read(QUEUE, [])


def reservations():
    return [e for e in entries() if e["stage"] not in ("完成", "失败", "已停止")]


def conflicts(host, gpus):
    return any(e["host"] == host and set(e["gpus"]) & set(gpus) for e in reservations())


def enqueue(recipe_id, run_ids):
    with store.lock(QUEUE.parent / "queue-lock"):
        pending = entries()
        if any(e["recipe_id"] == recipe_id for e in pending):
            return
        first = store.read(RUNS / run_ids[0] / "config.json")
        if conflicts(first["host"], first["gpus"]):
            raise ValueError("所选 GPU 已被配方队列预留")
        entry = dict(
            recipe_id=recipe_id,
            runs=run_ids,
            host=first["host"],
            gpus=first["gpus"],
            stage="排队中",
            active=None,
            created_at=store.now(),
        )
        pending.append(entry)
        store.write(QUEUE, pending)
        for rid in run_ids:
            store.write(RUNS / rid / "state.json", dict(stage="排队中"))


def _alive(run):
    info = store.read(run / "process.json", {})
    pid = info.get("pid")
    if type(pid) is not int:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        return b"job.py" in command and str(run).encode() in command
    except OSError:
        return False


def cancel_waiting(run):
    with store.lock(QUEUE.parent / "queue-lock"):
        state = store.read(run / "state.json", {})
        if state.get("stage") not in ("排队中", "数据已核验"):
            return False
        (run / "cancel.request").touch()
        store.write(run / "state.json", dict(stage="已停止"))
        return True


def tick(launch):
    with store.lock(QUEUE.parent / "queue-lock"):
        pending = entries()
        for entry in pending:
            if entry["stage"] in TERMINAL:
                continue
            active = entry.get("active")
            if active:
                run = RUNS / active
                stage = store.read(run / "state.json", {}).get("stage")
                if stage not in TERMINAL:
                    if not _alive(run) and stage != "恢复待确认":
                        store.write(
                            run / "state.json",
                            dict(
                                stage="恢复待确认",
                                error="监督进程已退出；为避免重复训练，请先停止该任务，再重新提交",
                            ),
                        )
                    continue
                entry["active"] = None
            next_run = next(
                (
                    RUNS / rid
                    for rid in entry["runs"]
                    if store.read(RUNS / rid / "state.json", {}).get("stage")
                    in ("排队中", "数据已核验")
                ),
                None,
            )
            if next_run is None:
                states = [
                    store.read(RUNS / rid / "state.json", {}).get("stage")
                    for rid in entry["runs"]
                ]
                entry["stage"] = (
                    "失败"
                    if "失败" in states
                    else "已停止"
                    if all(x == "已停止" for x in states)
                    else "完成"
                )
                continue
            if (next_run / "cancel.request").exists():
                store.write(next_run / "state.json", dict(stage="已停止"))
                continue
            cfg = store.read(next_run / "config.json", {})
            if cfg.get("dataset"):
                from .build import admission

                try:
                    batch = store.batch_path(cfg["dataset"])
                    with store.lock(batch):
                        summary = admission(batch)
                        snapshot = store.read(next_run / "mix.json", {}).get("qc", {})
                        if any(
                            summary.get(key) != snapshot.get(key)
                            for key in ("sample_sha256", "reviews_sha256")
                        ):
                            raise ValueError("质检快照不一致")
                except (ValueError, OSError):
                    store.write(
                        next_run / "state.json",
                        dict(
                            stage="失败",
                            error="质检结果已变化，请重新完成质检并提交配方",
                        ),
                    )
                    continue
            entry["active"] = next_run.name
            entry["stage"] = "运行中"
            store.write(QUEUE, pending)
            store.write(next_run / "state.json", dict(stage="启动中"))
            try:
                launch(next_run)
            except Exception as exc:
                store.write(next_run / "state.json", dict(stage="失败", error=str(exc)))
                entry["active"] = None
        store.write(QUEUE, pending)


def recover():
    """Recover a committed recipe if the server stopped between build and enqueue."""
    known = {rid for e in entries() for rid in e["runs"]}
    grouped = {}
    if not RUNS.exists():
        return
    for run in RUNS.iterdir():
        cfg = store.read(run / "config.json", {}) if run.is_dir() else {}
        if (
            cfg.get("dispatch_authorized")
            and cfg.get("recipe_id")
            and (
                store.read(
                    store.DATASETS
                    / cfg["dataset"]
                    / "recipes"
                    / (cfg["recipe_id"] + ".json"),
                    {},
                )
                or {}
            ).get("runs")
            and run.name not in known
            and store.read(run / "state.json", {}).get("stage") == "数据已核验"
        ):
            grouped.setdefault(cfg["recipe_id"], []).append(run.name)
    for recipe_id, rids in grouped.items():
        cfg = store.read(RUNS / rids[0] / "config.json")
        record = store.read(
            store.DATASETS / cfg["dataset"] / "recipes" / (recipe_id + ".json")
        )
        ordered = [rid for rid in record["runs"] if rid in rids]
        enqueue(recipe_id, ordered)


def start(launch, guard):
    def work():
        while True:
            try:
                with guard:
                    recover()
                    tick(launch)
            except Exception as exc:
                print("配方队列错误:", exc, flush=True)
            time.sleep(2)

    thread = threading.Thread(target=work, name="recipe-queue", daemon=True)
    thread.start()
    return thread
