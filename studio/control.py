"""Stop one experiment's verified supervisor and descendants on this host."""

import os
from pathlib import Path
import select
import signal
import sys
import time

from catalog import ROOT, read


def stop_tree(script, run):
    handles = {}

    def freeze(pid, verify=False):
        if pid in handles:
            return
        proc = Path("/proc") / str(pid)
        try:
            if proc.stat().st_uid != os.getuid():
                return
            fd = os.pidfd_open(pid)
        except (FileNotFoundError, ProcessLookupError):
            return
        try:
            if verify:
                argv = (proc / "cmdline").read_bytes().split(b"\0")
                args = argv[1:-1]
                if args[:1] == [b"-u"]:
                    args = args[1:]
                if (
                    len(args) != 2
                    or not args[0].endswith(os.fsencode("/" + script.name))
                    or not os.path.samefile(args[0], script)
                    or not os.path.samefile(args[1], run)
                ):
                    return
            signal.pidfd_send_signal(fd, signal.SIGSTOP)
            handles[pid] = fd
            # Wait until fork/exec is quiescent before walking child processes.
            for _ in range(200):
                state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if state in ("T", "t", "Z", "X"):
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError(f"进程 {pid} 尚未响应停止请求，请重试")
            for children in (proc / "task").glob("*/children"):
                for child in children.read_text().split():
                    freeze(int(child))
        except (FileNotFoundError, ProcessLookupError):
            pass
        finally:
            if pid not in handles:
                os.close(fd)

    def send(fd, sig):
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            pass

    try:
        for item in Path("/proc").iterdir():
            if item.name.isdecimal():
                freeze(int(item.name), True)
        for fd in reversed(handles.values()):
            send(fd, signal.SIGTERM)
            send(fd, signal.SIGCONT)
        alive = set(handles.values())
        deadline = time.monotonic() + 4
        while alive and time.monotonic() < deadline:
            alive.difference_update(select.select(list(alive), [], [], 0.2)[0])
        for fd in alive:
            send(fd, signal.SIGKILL)
        for fd in alive:
            if not select.select([fd], [], [], 3)[0]:
                raise RuntimeError("进程尚未退出，请重试停止")
    finally:
        for fd in handles.values():
            send(fd, signal.SIGCONT)
            os.close(fd)
    return len(handles)


if __name__ == "__main__":
    # Some training environments build Python without the Linux pidfd bindings.
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        system_python = "/usr/bin/python3"
        if os.path.samefile(sys.executable, system_python):
            raise SystemExit("系统 Python 缺少 pidfd 停止接口")
        os.execv(system_python, [system_python, *sys.argv])
    run = Path(sys.argv[1]).resolve()
    local_output = (
        run.parent == Path("/mnt/nvme0n1/wujinbo/wan-studio-runs")
        and read(run / "config.json").get("local_output") is True
    )
    if (run.parent != ROOT / "runs" and not local_output) or not (run / "cancel.request").exists():
        raise SystemExit("缺少该实验的停止请求")
    print(stop_tree(ROOT / "worker.py", run))
