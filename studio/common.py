"""Shared data helpers; importing this module never imports torch."""

from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import threading

from catalog import DATA


def source_group(row):
    name = row["id"].split("_scene4way_")[0]
    return re.sub(r"[_-]clip[_-]?\d+.*$", "", name, flags=re.I)


def action_words(caption):
    sentences = re.split(r"(?<=[.!?])\s+", caption.lower())
    # Cache captions normally put temporally ordered action details last.
    text = " ".join(sentences[-4:])
    stop = set(
        "the a his her she he then while with their and to into from of in on is it man woman subject main frame camera before after".split()
    )
    return set(re.findall(r"[a-z]{3,}|[\u4e00-\u9fff]", text)) - stop


_index_lock = threading.Lock()
_index_cache = None
_index_version = None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_version():
    path = Path(os.environ.get("WAN_INDEX", DATA / "index.jsonl.gz")).expanduser().resolve()
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def load_index_by_id():
    global _index_cache, _index_version
    with _index_lock:
        version = index_version()
        if version != _index_version:
            path = Path(version[0])
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as stream:
                _index_cache = {
                    r["id"]: r
                    for line in stream
                    if line.strip()
                    for r in [json.loads(line)]
                }
            _index_version = version
        return _index_cache


def trainable(row):
    shape = row.get("video_shape")
    return (
        isinstance(shape, list)
        and len(shape) == 4
        and all(type(x) is int for x in shape)
        and shape[0] == 16
        and 12 <= shape[1] <= 25
        and sorted(shape[2:]) == [60, 104]
        and type(row.get("num_frames")) is int
        and row["num_frames"] == 4 * shape[1] - 3
    )


def validate_cache(row):
    try:
        import torch

        data = torch.load(
            row["cache"], weights_only=True, mmap=True, map_location="cpu"
        )
        assert data["key"] == row["key"], "key 不一致"
        assert data["config_hash"] == row["config_hash"], "config_hash 不一致"
        assert isinstance(data["caption"], str) and data["caption"].strip(), (
            "caption 为空"
        )
        if "caption" in row:
            assert data["caption"].strip() == row["caption"].strip(), "caption 不一致"
        assert trainable(row), "索引尺寸不兼容训练器"
        assert list(data["video_latent"].shape) == row["video_shape"], (
            "latent 尺寸不一致"
        )
        assert data["metadata"]["fps"] == 24, "缓存 fps 须为 24"
        text = data["text_embedding"]
        assert text.ndim == 2 and text.shape[1] == 4096 and 0 < len(text) <= 512, (
            "文本嵌入尺寸错误"
        )
        assert (
            torch.isfinite(data["video_latent"]).all() and torch.isfinite(text).all()
        ), "缓存存在非有限值"
    except Exception as exc:
        return str(exc) or type(exc).__name__
    return None
