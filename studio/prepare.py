"""Create a source-disjoint holdout before constructing the training manifest."""

from collections import Counter, defaultdict, deque
from pathlib import Path
import random
import json
import sys
import torch
from catalog import DATA, ROOT, records, read, write


from common import source_group, action_words


def prepare(run):
    cfg = read(run / "config.json")
    category = next(
        c for c in read(DATA / "catalog.json") if c["id"] == cfg["category"]
    )
    rows = [r for r in records() if category["id"] in r["sampling"]["l2_ids"]]
    rng = random.Random(cfg["seed"])
    rng.shuffle(rows)
    candidates = []
    known = set()
    # Inspect actual caption text; keep repeated clips of the same original together.
    mapping = read(DATA / "parents.json")["mapping"]
    relevant = {a["name"] for a in category["actions"]}
    relevant.update(
        name for name, parent in mapping.items() if parent["l2_id"] == cfg["category"]
    )

    aliases = read(
        ROOT.parent.parent
        / "D_human_train/extensions/source_l2_sampling/configs/legacy_action_aliases.json"
    )["l3_to_l2"]
    relevant.update(
        name for name, parent in aliases.items() if parent == category["name"]
    )

    def class_actions(row):
        return relevant.intersection(row["sampling"].get("l3") or [])

    strata = defaultdict(list)
    for row in rows:
        strata[tuple(sorted(class_actions(row)))].append(row)
    pending = deque(iter(bucket) for bucket in strata.values())
    while pending:
        bucket = pending.popleft()
        r = next(bucket, None)
        if r is None:
            continue
        pending.append(bucket)
        group = source_group(r)
        if group in known:
            continue
        try:
            data = torch.load(
                r["cache"], weights_only=True, mmap=True, map_location="cpu"
            )
            caption = data["caption"].strip()
            if not caption or data["key"] != r["key"]:
                continue
        except (OSError, KeyError, RuntimeError):
            continue
        known.add(group)
        candidates.append((r, caption, action_words(caption)))
        if len(candidates) >= 160:
            break
    if len(candidates) < 10:
        raise ValueError(f"仅有 {len(candidates)} 组可读取的独立来源，无法留出 10 组")
    chosen = []
    labels = set()
    vocab = []
    pools = set()
    while len(chosen) < 10:

        def score(item):
            r, _, words = item
            actions = class_actions(r)
            distance = min(
                (1 - len(words & s) / max(1, len(words | s)) for s in vocab), default=1
            )
            return (
                bool(actions - labels),
                distance,
                r["sampling"]["pool"] not in pools,
            )

        pick = max(candidates, key=score)
        candidates.remove(pick)
        chosen.append(pick)
        labels.update(class_actions(pick[0]))
        vocab.append(pick[2])
        pools.add(pick[0]["sampling"]["pool"])
    groups = {source_group(r) for r, _, _ in chosen}
    training = [r for r in rows if source_group(r) not in groups]
    if cfg.get("limit"):
        training = training[: cfg["limit"]]
    if not training:
        raise ValueError("留出来源后无训练数据")
    with (run / "train.jsonl").open("w") as f:
        for r in training:
            print(json.dumps(r, ensure_ascii=False), file=f)
    cases = []
    for r, caption, _ in chosen:
        cases.append(
            dict(
                kind="留出样本",
                name=r["id"],
                prompt=caption,
                labels=r["sampling"].get("l3") or [],
                source=source_group(r),
                shape=r["video_shape"],
                cache=r["cache"],
            )
        )
    actions = category["actions"][:]
    rng.shuffle(actions)
    # Include both framings when the selected L2 has both.
    order = []
    for frame in ("全身", "半身"):
        first = next((a for a in actions if a["framing"] == frame), None)
        if first:
            order.append(first)
    order.extend(a for a in actions if a not in order)
    for a in order[: cfg["products"]]:
        cases.append(dict(a, kind="产品描述", shape=[16, 20, 60, 104]))
    for i, c in enumerate(cases):
        c["seed"] = cfg["seed"] + i
    write(run / "cases.json", cases)
    write(
        run / "split.json",
        dict(
            matched=len(rows),
            train=len(training),
            heldout=10,
            groups_excluded=len(rows)
            - sum(source_group(r) not in groups for r in rows),
            explicit_actions=sorted(labels),
            sources=dict(Counter(r["sampling"]["pool"] for r in training)),
        ),
    )
    print("split", read(run / "split.json"), flush=True)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
