"""Append-only human review log and explicit training admission gate."""

from collections import Counter
from pathlib import Path
import json
import math
import os
from common import sha256_file
from . import store

ISSUES = {
    "caption_mismatch",
    "label_wrong",
    "quality",
    "multi_person",
    "watermark",
    "cut",
    "low_motion",
    "bad_aspect",
    "other",
}


def latest(batch):
    path = Path(batch) / "qc/reviews.jsonl"
    result = {}
    if path.exists():
        for row in store.jsonl(path):
            result[row["idx"]] = row
    return result


def save(batch_dir, review):
    batch = Path(batch_dir)
    if not isinstance(review, dict):
        raise ValueError("审核记录须为对象")
    with store.lock(batch):
        sample = store.read(batch / "qc/sample.json")
        idx = review.get("idx")
        if type(idx) is not int or not sample or not 0 <= idx < len(sample["items"]):
            raise ValueError("样本索引不存在")
        item = sample["items"][idx]
        if review.get("id") != item["id"]:
            raise ValueError("样本 ID 不匹配，请刷新页面")
        verdict = review.get("verdict")
        if verdict not in ("pass", "fail"):
            raise ValueError("审核结果须为 pass 或 fail")
        annotator = review.get("annotator")
        if (
            not isinstance(annotator, str)
            or not annotator.strip()
            or len(annotator) > 100
        ):
            raise ValueError("请填写审核人姓名（最多 100 字）")
        note = review.get("note", "")
        if not isinstance(note, str) or len(note) > 10000:
            raise ValueError("备注须为字符串，最多 10000 字")
        stats = store.read(batch / "stats.json")
        valid_l2 = {c["id"] for c in stats["categories"]["l2"]}
        values = {}
        for field, allowed in [
            ("issues", ISSUES),
            ("wrong_l2", {x["id"] for x in item["l2"]}),
            ("missing_l2", valid_l2 - {x["id"] for x in item["l2"]}),
        ]:
            v = review.get(field, [])
            if not isinstance(v, list) or any(
                not isinstance(x, str) or x not in allowed for x in v
            ):
                raise ValueError(field + " 含无效选项")
            values[field] = sorted(set(v))
        row = dict(
            idx=idx,
            id=item["id"],
            annotator=annotator.strip(),
            saved_at=store.now(),
            verdict=verdict,
            note=note,
            schema_version=1,
            **values,
        )
        path = batch / "qc/reviews.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        store.invalidate_qc(batch)
        store.state(batch, "质检中")
        return dict(ok=True, reviewed=len(latest(batch)))


def finish(batch_dir, override_reason=None):
    batch = Path(batch_dir)
    if override_reason is not None and (
        not isinstance(override_reason, str)
        or not override_reason.strip()
        or len(override_reason) > 2000
    ):
        raise ValueError("放行理由须为 1–2000 字")
    try:
        threshold = float(os.environ.get("WAN_QC_THRESHOLD", ".9"))
    except ValueError:
        raise ValueError("WAN_QC_THRESHOLD 须在 0–1 内")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("WAN_QC_THRESHOLD 须在 0–1 内")
    with store.lock(batch):
        sample = store.read(batch / "qc/sample.json", {}).get("items", [])
        reviews = latest(batch)
        if not sample or any(
            i["idx"] not in reviews or reviews[i["idx"]]["id"] != i["id"]
            for i in sample
        ):
            raise ValueError("必须全部审完才能完成质检")
        active = [reviews[i["idx"]] for i in sample]
        passed = sum(r["verdict"] == "pass" for r in active)
        rate = passed / len(sample)
        failed = {r["id"] for r in active if r["verdict"] == "fail"}
        gate = "可训练" if rate >= threshold or override_reason else "待处理"
        summary = dict(
            total=len(sample),
            reviewed=len(active),
            **{"pass": passed, "fail": len(sample) - passed},
            pass_rate=rate,
            threshold=threshold,
            issues=dict(Counter(x for r in active for x in r["issues"])),
            fail_by_l2=dict(
                Counter(c["id"] for i in sample if i["id"] in failed for c in i["l2"])
            ),
            wrong_l2=dict(Counter(x for r in active for x in r["wrong_l2"])),
            missing_l2=dict(Counter(x for r in active for x in r["missing_l2"])),
            excluded_ids=sorted(failed),
            gate=gate,
            override_reason=override_reason.strip() if override_reason else None,
            finished_at=store.now(),
            sample_sha256=sha256_file(batch / "qc/sample.json"),
            reviews_sha256=sha256_file(batch / "qc/reviews.jsonl"),
        )
        store.write(batch / "qc/summary.json", summary)
        store.state(batch, gate)
        return summary
