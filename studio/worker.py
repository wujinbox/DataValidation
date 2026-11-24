"""Remote job supervisor. GPU locks live on the compute host."""

import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

from catalog import ROOT, read, write
from cluster import GPU_QUERY, parse_gpu


def worker(run):
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=64x64:d=0.1",
            "-vf",
            "drawtext=text=Base:fontsize=12",
            "-c:v",
            "libx264",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    cfg = read(run / "config.json")
    if (run / "execution.json").exists():
        cfg.update(read(run / "execution.json"))
    locks = []
    for i in sorted(cfg["gpus"]):
        lock = open(f"/tmp/wan-l2-{os.getuid()}-gpu{i}.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    available = {
        c["index"]
        for c in parse_gpu(subprocess.check_output(GPU_QUERY, text=True))
        if c["idle"]
    }
    if not set(cfg["gpus"]) <= available:
        raise RuntimeError("所选 GPU 已被占用，请重新选卡")
    if shutil.disk_usage(run).free < 7 * 2**30:
        raise RuntimeError("实验目录可用空间不足 7 GiB，无法保存模型和视频")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(map(str, cfg["gpus"])),
        OMP_NUM_THREADS="2",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTHONUNBUFFERED="1",
    )

    def stage(name):
        if (run / "cancel.request").exists():
            raise RuntimeError("实验已请求停止")
        write(run / "state.json", dict(stage=name))

    count = len(cfg["gpus"])
    if not (run / "transformer").exists():
        stage("准备测试集")
        if cfg.get("prepared_data"):
            for filename in ("train.jsonl", "cases.json", "split.json"):
                if not (run / filename).is_file():
                    raise FileNotFoundError(run / filename)
        else:
            subprocess.run(
                [sys.executable, str(ROOT / "prepare.py"), str(run)],
                env=env,
                check=True,
            )
        stage("SFT 训练")
        train = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={count}",
            str(ROOT.parent / "train_cached_sft.py"),
            "--model",
            str(ROOT.parent / "outputs/l2-sport-fullbody/base"),
            "--index",
            str(run / "train.jsonl"),
            "--limit",
            "0",
            "--frames",
            "0",
            "--batch",
            str(cfg["batch"]),
            "--lr",
            str(cfg["lr"]),
            "--seed",
            str(cfg["seed"]),
            "--warmup",
            "0",
            "--save",
            str(run / "transformer"),
        ]
        train += ["--steps", str(cfg["steps"])] if cfg["steps"] else ["--one-epoch"]
        subprocess.run(train, env=env, check=True)
    stage("前后对照评测")
    cmd = [sys.executable, str(ROOT / "evaluate.py"), str(run)]
    subprocess.run(cmd + ["encode"], env=env, check=True)
    children = []
    try:
        for rank, i in enumerate(cfg["gpus"]):
            children.append(
                subprocess.Popen(
                    cmd + [str(rank), str(count)],
                    env=dict(env, CUDA_VISIBLE_DEVICES=str(i)),
                )
            )
        for child in children:
            if child.wait():
                raise RuntimeError("对照视频生成失败，详见日志")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
    write(run / "evaluated.json", dict(cases=len(read(run / "cases.json"))))


if __name__ == "__main__":
    worker(Path(sys.argv[1]).resolve())
