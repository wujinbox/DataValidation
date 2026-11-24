"""Source-disjoint, category-covering evaluation case selection."""

from collections import Counter, defaultdict
import random
from common import action_words, source_group


def select(rows, target_l2, per_l2, max_cases, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("source_group") or source_group(row)].append(row)
    representatives = []
    group_labels = {g: {x for r in rs for x in r["l2_ids"]} for g, rs in groups.items()}
    remaining = Counter(x for labels in group_labels.values() for x in labels)
    for group, bucket in sorted(groups.items()):
        lengths = sorted(len(r.get("caption", "").split()) for r in bucket)
        median = lengths[len(lengths) // 2]
        row = min(
            bucket,
            key=lambda r: (
                r.get("review_required", False),
                abs(len(r.get("caption", "").split()) - median),
                r["id"],
            ),
        )
        representatives.append((group, row))
    counts = Counter()
    words = []
    people = set()
    selected = []
    targets = set(target_l2)
    while (
        representatives
        and len(selected) < max_cases
        and any(counts[x] < per_l2 for x in targets)
    ):
        choices = []
        for group, row in representatives:
            # Never hold out the last source containing any participating label.
            if any(remaining[x] <= 1 for x in group_labels[group] & targets):
                continue
            labels = set(row["l2_ids"]) & targets
            gain = sum(counts[x] < per_l2 for x in labels)
            if not gain:
                continue
            vocab = action_words(row.get("caption", ""))
            distance = min(
                (1 - len(vocab & w) / max(1, len(vocab | w)) for w in words), default=1
            )
            score = (
                gain,
                distance,
                not row.get("ip") or row["ip"] not in people,
                rng.random(),
            )
            choices.append((score, group, row, vocab))
        if not choices:
            break
        _, group, row, vocab = max(choices, key=lambda x: x[0])
        representatives = [x for x in representatives if x[0] != group]
        counts.update(set(row["l2_ids"]) & targets)
        remaining.subtract(group_labels[group])
        words.append(vocab)
        people.add(row.get("ip"))
        selected.append(
            dict(
                kind="留出样本",
                name=row["id"],
                prompt=row.get("caption", ""),
                labels=row["l2_ids"],
                covered_l2=sorted(set(row["l2_ids"]) & targets),
                source=group,
                shape=row["video_shape"],
                cache=row["cache"],
                seed=seed + len(selected),
            )
        )
    return selected


def products(catalog, target_l2, counts, limit, seed, mixed=False):
    if not limit:
        return []
    rng = random.Random(seed)
    result = []
    byid = {x["id"]: x for x in catalog}
    for cid in (
        sorted(target_l2, key=lambda x: (-counts.get(x, 0), x)) if mixed else target_l2
    ):
        actions = byid.get(cid, {}).get("actions", [])[:]
        rng.shuffle(actions)
        order = []
        for frame in ("全身", "半身"):
            first = next((x for x in actions if x.get("framing") == frame), None)
            if first:
                order.append(first)
        order.extend(x for x in actions if x not in order)
        for action in order[: 1 if mixed else limit]:
            result.append(
                dict(
                    action,
                    kind="产品描述",
                    shape=[16, 20, 60, 104],
                    covered_l2=[cid],
                    source="catalog:" + cid,
                )
            )
            if len(result) >= limit:
                return result
    return result if limit else []


def manifest(cases, target_l2, truncated=None):
    coverage = {
        x: [
            c["name"]
            for c in cases
            if x in c.get("covered_l2", []) and c["kind"] == "留出样本"
        ]
        for x in target_l2
    }
    return [
        dict(
            name=c["name"],
            kind=c["kind"],
            prompt=c["prompt"],
            covered_l2=c.get("covered_l2", []),
            source=c.get("source"),
        )
        for c in cases
    ] + [
        dict(
            coverage=dict(
                l2=coverage,
                uncovered=[x for x, v in coverage.items() if not v],
                truncated=truncated or [],
            )
        )
    ]
