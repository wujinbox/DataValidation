from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import subprocess


PYTHON = "/home/zkln/miniconda3/envs/fastvideo/bin/python"
REMOTE = "/home/wujinbo/008/mnt/nvme0n1/wujinbo/code/Wan1.3B/studio"
HOSTS = {
    f"{n:04}": f"token-zkln-{'gpu-' if n >= 35 else ''}h100-{n:04}.host.platform.shaipower.com"
    for n in range(8, 44)
}
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def ssh(host, argv):
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=4",
        "-o",
        "ServerAliveInterval=20",
        "wujinbo@" + HOSTS[host],
        shlex.join(argv),
    ]


def parse_gpu(text):
    cards = []
    for line in text.strip().splitlines():
        i, used, total, util = map(int, line.split(","))
        cards.append(dict(index=i, used=used, total=total, util=util, idle=used < 64 and util == 0))
    return cards


def probe(host):
    check = (
        f"test -x {PYTHON} && test -r {REMOTE}/data/catalog.json && test -r /mnt/nvme0n1/models/Wan2.2-S2V-14B/Wan2.1_VAE.pth && test -r /mnt/nvme0n1/models/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth && "
        + shlex.join(GPU_QUERY)
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
