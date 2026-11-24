"""Read rank-zero metrics and completed comparison cases from the experiment log."""

from collections import deque
from threading import Lock
import re
from catalog import read

_cache = {}
_lock = Lock()
step = re.compile(rb"^step=(\d+) seconds=([\d.]+) loss=([\d.]+)")
total = re.compile(rb"^records=\d+ steps=(\d+)")
case = re.compile(rb"^sft case=(\d+)/")


def progress(run, cfg):
    path = run / "job.log"
    with _lock:
        pos, budget, done, times, loss, pairs = _cache.get(
            run, (0, cfg["steps"], 0, deque(maxlen=20), None, set())
        )
        if path.exists():
            with path.open("rb") as f:
                f.seek(pos)
                for line in f:
                    if not line.endswith(b"\n"):
                        break
                    pos += len(line)
                    m = step.match(line)
                    if m:
                        done = int(m[1])
                        times.append(float(m[2]))
                        loss = float(m[3])
                    elif m := total.match(line):
                        budget = int(m[1])
                    elif m := case.match(line):
                        pairs.add(int(m[1]))
        _cache[run] = pos, budget, done, times, loss, pairs
        try:
            cases = len(read(run / "cases.json"))
        except (OSError, ValueError, TypeError):
            cases = 10 + cfg["products"]
        return dict(
            step=done,
            total=budget,
            seconds=sum(times) / len(times) if times else None,
            loss=loss,
            pairs=len(pairs),
            cases=cases,
        )
