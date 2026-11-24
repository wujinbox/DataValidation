"""Deterministic multi-label sampling with source/IP diversity."""

from collections import Counter
from pathlib import Path
import math
import random
from . import store


def _person(r):
    return r.get("ip") or "unknown:" + r["id"]


def _item(row, idx, reason):
    return dict(
        idx=idx,
        id=row["id"],
        video_path=row.get("video_path"),
        caption=row.get("caption", ""),
        static_caption=row.get("static_caption"),
        dynamic_caption=row.get("dynamic_caption"),
        l1=[
            dict(id=k, name=n)
            for k, n in zip(row.get("l1_ids", []), row.get("l1_names", []))
        ],
        l2=[
            dict(
                id=k,
                name=(row.get("l2_names") or [])[i]
                if i < len(row.get("l2_names") or [])
                else k,
            )
            for i, k in enumerate(row["l2_ids"])
        ],
        body_framing=row.get("body_framing"),
        orientation=row.get("orientation"),
        width=row.get("width"),
        height=row.get("height"),
        duration=row.get("duration"),
        fps=row.get("fps"),
        global_ip_id=row.get("ip"),
        review_required=row.get("review_required", False),
        language_review=row.get("language_review", []),
        reason=reason,
        source_group=row["source_group"],
    )


def sample(batch_dir, n=20, seed=42, extra_l2=None, k_per_l2=3):
    batch = Path(batch_dir)
    if (
        type(n) is not int
        or not 1 <= n <= 1000
        or type(seed) is not int
        or not 0 <= seed <= 2147483647
    ):
        raise ValueError("抽样数量须为 1–1000，seed 须为有效非负整数")
    if type(k_per_l2) is not int or not 1 <= k_per_l2 <= 100:
        raise ValueError("每类抽检数量须为 1–100")
    with store.lock(batch):
        rows = list(store.jsonl(batch / "pool.jsonl"))
        old = store.read(batch / "qc/sample.json")
        if not rows:
            raise ValueError("训练池为空，无法抽样")
        byid = {r["id"]: r for r in rows}
        total = Counter(x for r in rows for x in r["l2_ids"])
        if extra_l2 is not None:
            if (
                not isinstance(extra_l2, list)
                or not extra_l2
                or any(not isinstance(x, str) or x not in total for x in extra_l2)
            ):
                raise ValueError("类别抽检须指定池中存在的 L2")
            if not old:
                raise ValueError("请先完成基础抽样")
            chosen = [byid[x["id"]] for x in old["items"]]
            result_items = old["items"][:]
            rng = random.Random(old["seed"] + len(chosen))
        else:
            chosen = []
            result_items = []
            rng = random.Random(seed)
        denominator = sum(total.values()) or 1
        targets = {k: n * v / denominator for k, v in total.items()}
        quota = {k: math.floor(v) for k, v in targets.items()}
        for k in sorted(targets, key=lambda k: (-(targets[k] - quota[k]), k))[
            : n - sum(quota.values())
        ]:
            quota[k] += 1
        population_frame = Counter(r.get("body_framing") for r in rows)
        population_orientation = Counter(r.get("orientation") for r in rows)
        covered = Counter(x for r in chosen for x in r["l2_ids"])
        warnings = []

        def eligible(row, current):
            return (
                row["id"] not in {x["id"] for x in current}
                and _person(row) not in {_person(x) for x in current}
                and row["source_group"] not in {x["source_group"] for x in current}
            )

        def pick(current, candidates):
            people = {_person(x) for x in current}
            groups = {x["source_group"] for x in current}
            choices = [
                r
                for r in candidates
                if _person(r) not in people and r["source_group"] not in groups
            ]
            if len(choices) > 20000:
                choices = rng.sample(choices, 20000)
            frames = Counter(r.get("body_framing") for r in current)
            orientations = Counter(r.get("orientation") for r in current)
            size = len(current) + 1

            def score(row):
                distance = 0
                for counts, pop, key in [
                    (frames, population_frame, "body_framing"),
                    (orientations, population_orientation, "orientation"),
                ]:
                    for value, count in pop.items():
                        distance += abs(
                            (counts[value] + (row.get(key) == value)) / size
                            - count / len(rows)
                        )
                return (
                    sum(covered[x] < quota[x] for x in row["l2_ids"]),
                    sum(covered[x] == 0 for x in row["l2_ids"]),
                    -distance,
                    rng.random(),
                )

            return max(choices, key=score) if choices else None

        if extra_l2 is None:
            for _ in range(min(n, len(rows))):
                row = pick(chosen, rows)
                if row is None:
                    break
                new = [x for x in row["l2_ids"] if not covered[x]]
                chosen.append(row)
                covered.update(row["l2_ids"])
                result_items.append(
                    _item(
                        row,
                        len(result_items),
                        "覆盖新 L2: " + ", ".join(new) if new else "分布平衡抽样",
                    )
                )
            if any(r.get("review_required") for r in rows) and not any(
                r.get("review_required") for r in chosen
            ):
                # Last selected has the lowest remaining marginal coverage priority.
                for pos in range(len(chosen) - 1, -1, -1):
                    remaining = chosen[:pos] + chosen[pos + 1 :]
                    candidates = [
                        r
                        for r in rows
                        if r.get("review_required") and eligible(r, remaining)
                    ]
                    if candidates:
                        replacement = pick(remaining, candidates)
                        chosen[pos] = replacement
                        result_items[pos] = _item(
                            replacement, pos, "review_required 保底抽检"
                        )
                        break
                else:
                    warnings.append("IP／源组约束下无法加入 review_required 样本")
            if len(chosen) < n:
                warnings.append(f"独立 IP／源组不足，仅抽取 {len(chosen)} 条")
        else:
            for cid in dict.fromkeys(extra_l2):
                for _ in range(k_per_l2):
                    row = pick(chosen, [r for r in rows if cid in r["l2_ids"]])
                    if row is None:
                        warnings.append(cid + " 独立 IP／源组不足，未达到追加配额")
                        break
                    chosen.append(row)
                    covered.update(row["l2_ids"])
                    result_items.append(
                        _item(row, len(result_items), "类别抽检: " + cid)
                    )
        cover = sorted({x for r in chosen for x in r["l2_ids"]})
        result = dict(
            seed=old["seed"] if extra_l2 is not None else seed,
            n=len(result_items),
            created_at=old["created_at"]
            if old
            else store.read(batch / "source.json")["registered_at"],
            items=result_items,
            coverage=dict(
                covered_l2=cover,
                uncovered_l2=sorted(set(total) - set(cover)),
                framing=dict(Counter(r.get("body_framing") or "null" for r in chosen)),
                orientation=dict(
                    Counter(r.get("orientation") or "null" for r in chosen)
                ),
            ),
            warnings=warnings,
        )
        if old and result == old:
            return old
        # reviews is JSONL, so only its existence is needed.
        if old and extra_l2 is None and (batch / "qc/reviews.jsonl").exists():
            raise ValueError("已有审核记录，不能重新抽样；请使用类别抽检追加")
        store.write(batch / "qc/sample.json", result)
        store.invalidate_qc(batch)
        store.state(
            batch, "质检中" if (batch / "qc/reviews.jsonl").exists() else "待质检"
        )
        return result
