"""Atomic artifacts and process/thread-safe batch transactions."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import fcntl
import json
import os
import re
import tempfile
import threading

from catalog import ROOT
from common import sha256_file

DATASETS = ROOT / "datasets"
ID = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
_mutexes = {}
_mutex_guard = threading.Lock()


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def batch_path(batch_id):
    if (
        not isinstance(batch_id, str)
        or not ID.fullmatch(batch_id)
        or batch_id in (".", "..")
    ):
        raise ValueError("无效的批次 ID")
    path = DATASETS / batch_id
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError("批次不存在")
    return path


@contextmanager
def lock(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    name = str(path.resolve())
    with _mutex_guard:
        mutex = _mutexes.setdefault(name, threading.Lock())
    with mutex, (path / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def atomic_stream(path, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(
            fd, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})
        ) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write(path, data):
    with atomic_stream(path) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with atomic_stream(path) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def state(batch, stage, **fields):
    value = dict(stage=stage, updated_at=now(), **fields)
    write(Path(batch) / "state.json", value)
    return value


def invalidate_qc(batch):
    (Path(batch) / "qc/summary.json").unlink(missing_ok=True)


def register(path, batch_id=None):
    source = Path(path)
    if (
        not source.is_absolute()
        or source.suffix != ".jsonl"
        or not source.is_file()
        or not os.access(source, os.R_OK)
    ):
        raise ValueError("请提供可读取的绝对路径 .jsonl 文件")
    source = source.resolve()
    before = source.stat()
    digest = sha256_file(source)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("源文件在登记期间发生变化，请稍后重试")
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", source.stem)[:110] or "dataset"
    name = batch_id or stem + "-" + digest[:8]
    if not isinstance(name, str) or not ID.fullmatch(name) or name in (".", ".."):
        raise ValueError("无效的批次 ID")
    with lock(DATASETS):
        for existing in DATASETS.iterdir():
            if (
                existing.is_dir()
                and read(existing / "source.json", {}).get("sha256") == digest
            ):
                return existing, False
        batch = DATASETS / name
        if batch.exists():
            raise ValueError("批次 ID 已被其他文件使用")
        batch.mkdir()
        write(
            batch / "source.json",
            dict(
                path=str(source),
                sha256=digest,
                size=after.st_size,
                mtime_ns=after.st_mtime_ns,
                rows=0,
                registered_at=now(),
                registered_by=os.environ.get("USER", "local"),
            ),
        )
        state(batch, "登记")
    return batch, True


def detail(batch):
    from .qc import latest

    return dict(
        source=read(batch / "source.json"),
        state=read(batch / "state.json"),
        stats=read(batch / "stats.json"),
        sample=read(batch / "qc/sample.json"),
        reviews=list(latest(batch).values()),
        summary=read(batch / "qc/summary.json"),
    )


def datasets():
    if not DATASETS.exists():
        return []
    result = []
    for batch in sorted(DATASETS.iterdir(), reverse=True):
        source = read(batch / "source.json") if batch.is_dir() else None
        if source:
            summary = read(batch / "qc/summary.json", {})
            result.append(
                dict(
                    batch_id=batch.name,
                    source=source["path"],
                    rows=source["rows"],
                    stage=read(batch / "state.json", {}).get("stage", "失败"),
                    pass_rate=summary.get("pass_rate"),
                )
            )
    return result
