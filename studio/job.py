import fcntl
from pathlib import Path
import subprocess
import sys
from catalog import read, write
from cluster import PUBLISH, PYTHON, REMOTE, remote_run, ssh
from publish import publish


def run_job(run):
    lock = (run / "job.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = read(run / "config.json")
    if (run / "execution.json").exists():
        cfg.update(read(run / "execution.json"))
    try:
        if (run / "cancel.request").exists():
            return
        if not (run / "evaluated.json").exists():
            subprocess.run(
                ssh(
                    cfg["host"],
                    [PYTHON, REMOTE + "/worker.py", remote_run(run.name, cfg)],
                ),
                check=True,
            )
        if (run / "cancel.request").exists():
            return
        if not PUBLISH:
            write(run / "state.json", dict(stage="完成"))
            return
        write(run / "state.json", dict(stage="飞书归档"))
        url = publish(run)
        write(run / "state.json", dict(stage="完成", url=url))
    except Exception as exc:
        write(run / "state.json", dict(stage="失败", error=str(exc)))
        raise


if __name__ == "__main__":
    run_job(Path(sys.argv[1]).resolve())
