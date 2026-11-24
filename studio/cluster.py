from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from catalog import ROOT

PYTHON = os.environ.get("WAN_STUDIO_REMOTE_PYTHON", sys.executable)
DATA_PYTHON = os.environ.get("WAN_STUDIO_DATA_PYTHON", sys.executable)
REMOTE = os.environ.get("WAN_STUDIO_REMOTE_DIR", str(ROOT)).rstrip("/")
REMOTE_OUTPUT = os.environ.get("WAN_STUDIO_REMOTE_OUTPUT", REMOTE + "/runs")
HOSTS = json.loads(os.environ.get("WAN_STUDIO_HOSTS", "{}"))
if not isinstance(HOSTS, dict) or any(
    not isinstance(target, str) or not target or target.startswith("-")
    for target in HOSTS.values()
):
    raise ValueError("WAN_STUDIO_HOSTS 必须是 GPU 节点编号到 SSH 目标的 JSON 对象")

BASE = Path(
    os.environ.get("WAN_BASE_MODEL", ROOT.parent / "outputs/l2-sport-fullbody/base")
).expanduser()
ENC = Path(
    os.environ.get("WAN_ENCODER_DIR", ROOT.parent / "models/Wan2.2-S2V-14B")
).expanduser()
NEGATIVE = Path(
    os.environ.get("WAN_NEGATIVE_EMBEDDING", ROOT.parent / "outputs/l2-sport-fullbody/ab/negative.pt")
).expanduser()
PUBLISH = bool(os.environ.get("WAN_FEISHU_SPACE") and os.environ.get("WAN_FEISHU_PARENT"))
REMOTE_ENV = (
    "WAN_ACTION_ALIASES",
    "WAN_BASE_MODEL",
    "WAN_DATA_DIR",
    "WAN_ENCODER_DIR",
    "WAN_INDEX",
    "WAN_NEGATIVE_EMBEDDING",
    "WAN_STUDIO_REMOTE_OUTPUT",
    "WAN_WAN_MODULES",
)
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def ssh(host, argv):
    settings = [f"{key}={os.environ[key]}" for key in REMOTE_ENV if key in os.environ]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=4",
        "-o",
        "ServerAliveInterval=20",
        HOSTS[host],
        shlex.join(["env", *settings, *argv]),
    ]


def remote_run(name, config):
    directory = REMOTE_OUTPUT if config.get("local_output") else REMOTE + "/runs"
    return directory.rstrip("/") + "/" + name


def parse_gpu(text):
    cards = []
    for line in text.strip().splitlines():
        i, used, total, util = map(int, line.split(","))
        cards.append(dict(index=i, used=used, total=total, util=util, idle=used < 64 and util == 0))
    return cards


def probe(host):
    encoder_dir = Path(
        os.environ.get("WAN_ENCODER_DIR", Path(REMOTE).parent / "models/Wan2.2-S2V-14B")
    ).expanduser()
    check = " && ".join(
        [
            shlex.join(["test", "-x", PYTHON]),
            shlex.join(["test", "-r", REMOTE + "/worker.py"]),
            shlex.join(["test", "-r", str(encoder_dir / "Wan2.1_VAE.pth")]),
            shlex.join(GPU_QUERY),
        ]
    )
    try:
        result = subprocess.run(
            ssh(host, ["bash", "-c", check]), capture_output=True, text=True, check=True, timeout=9
        )
        return dict(host=host, cards=parse_gpu(result.stdout))
    except (ValueError, subprocess.SubprocessError):
        return dict(host=host, cards=[], unavailable=True)


def inventory():
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(probe, HOSTS))


if __name__ == "__main__":
    print(json.dumps(inventory()))
