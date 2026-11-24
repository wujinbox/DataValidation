"""Deterministic planning and atomic materialization of prepared runs."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import csv
import io
import math
import random
import secrets
import shutil

from catalog import DATA, ROOT
from common import (
    index_version,
    load_index_by_id,
    sha256_file,
    trainable,
    validate_cache,
)
from . import store, cases
from .recipe import validate

RUNS = ROOT / "runs"


@dataclass
class RunPlan:
    category: str
    name: str
    targets: list
    rows: list
    matched: int
    cases: list
    heldout_groups: set
    before_filter: Counter
    after_filter: Counter
    before_mix: int
    after_mix: int
    recipe: dict
    warnings: list = field(default_factory=list)
    mix_counts: dict = field(default_factory=dict)

    def preview(self):
        t = self.recipe["train"]
        counts = Counter(tuple(r["video_shape"]) for r in self.rows)
        epoch_steps = sum(n // t["batch"] for n in counts.values()) // len(t["gpus"])
        steps = t["steps"] or epoch_steps
        heldout = sum(c["kind"] == "留出样本" for c in self.cases)
        # Training benchmark is 3–4 sec/sample/H100; evaluation is approximate.
        train_seconds = steps * t["batch"] * 3.5
        eval_seconds = estimate_eval(
            self.cases, self.recipe["eval"]["sampling_steps"], len(t["gpus"])
        )
        return dict(
            category=self.category,
            name=self.name,
            train_rows=len(self.rows),
            unique_rows=len({r["id"] for r in self.rows}),
            heldout=heldout,
            products=len(self.cases) - heldout,
            cases=len(self.cases),
            categories=self.targets,
            steps=steps,
            epoch_steps=epoch_steps,
            train_seconds=round(train_seconds),
            eval_seconds=round(eval_seconds),
            estimated_seconds=round(train_seconds + eval_seconds),
            estimate_note="训练按3–4秒/样本/H100；评测粗估每对120秒（77帧40步），不含编码、加载与归档",
            warnings=self.warnings,
        )


def estimate_eval(selected, steps, gpus):
    # Paired evaluation has two versions and CFG doubles each denoising forward pass.
    # Explicitly approximate 120 sec/pair @40 steps.
    seconds = 120.0
    return (
        sum(
            seconds * (max(1, (c["shape"][1] * 4 - 3)) / 77) * (steps / 40)
            for c in selected
        )
        / gpus
    )


def admission(batch):
    summary = store.read(batch / "qc/summary.json")
    if not summary or summary.get("gate") != "可训练":
        raise ValueError("批次质检尚未准入，不能构建训练集")
    for name, key in [
        ("sample.json", "sample_sha256"),
        ("reviews.jsonl", "reviews_sha256"),
    ]:
        if summary.get(key) != sha256_file(batch / "qc" / name):
            raise ValueError("质检记录已变化，请重新完成质检")
    source = store.read(batch / "source.json")
    stats = store.read(batch / "stats.json")
    stat = Path(source["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (source["size"], source["mtime_ns"]):
        raise ValueError("源数据已变化，请重新登记")
    if (
        stats["meta"]["sha256"] != source["sha256"]
        or list(index_version()) != stats["meta"]["index_version"]
    ):
        raise ValueError("源数据或缓存索引版本已变化，请重新统计新的批次")
    return summary


def _filtered(rows, filters, seed, excluded):
    result = []
    for r in rows:
        if r["id"] in excluded:
            continue
        if filters["min_short_side"] and (
            not r.get("width")
            or not r.get("height")
            or min(r["width"], r["height"]) < filters["min_short_side"]
        ):
            continue
        if (
            r.get("body_framing") not in filters["body_framing"]
            or r.get("orientation") not in filters["orientation"]
        ):
            continue
        if (
            filters["exclude_review_required"]
            and r.get("review_required")
            or filters["exclude_body_framing_mixed"]
            and r.get("body_framing_mixed")
        ):
            continue
        if (
            filters["dataset_source"]
            and r.get("dataset_source") not in filters["dataset_source"]
        ):
            continue
        result.append(r)
    if filters["max_per_ip"]:
        rng = random.Random(seed)
        groups = defaultdict(list)
        for r in result:
            groups[r.get("ip") or "unknown:" + r["id"]].append(r)
        allowed = set()
        for _, bucket in sorted(groups.items()):
            rng.shuffle(bucket)
            allowed.update(r["id"] for r in bucket[: filters["max_per_ip"]])
        result = [r for r in result if r["id"] in allowed]
    return result


def _weighted(rows, targets, mix, target_size, seed):
    rng = random.Random(seed)
    weighted = []
    for r in rows:
        weight = max(
            (mix.get(x, mix["default"]) for x in set(r["l2_ids"]) & set(targets)),
            default=0,
        )
        repetitions = math.floor(weight) + int(rng.random() < weight % 1)
        weighted.extend([r] * repetitions)
    if target_size and weighted:
        if target_size <= len(weighted):
            weighted = rng.sample(weighted, target_size)
        else:
            repeats, remainder = divmod(target_size, len(weighted))
            weighted = weighted * repeats + rng.sample(weighted, remainder)
    return weighted


def plan(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("配方须为 JSON 对象")
    batch = store.batch_path(recipe.get("dataset"))
    with store.lock(batch):
        return _plan(recipe, batch)


def _plan(recipe, batch, invalid_ids=None):
    stats = store.read(batch / "stats.json")
    r, warnings = validate(recipe, stats)
    summary = admission(batch)
    rows = list(store.jsonl(batch / "pool.jsonl"))
    before = Counter(x for row in rows for x in row["l2_ids"])
    rows = _filtered(
        rows,
        r["filters"],
        r["train"]["seed"],
        set(summary["excluded_ids"]) | set(invalid_ids or []),
    )
    index = load_index_by_id()
    # Dataset taxonomy always takes precedence over index.sampling taxonomy.
    joined = []
    for row in rows:
        cached = index.get(row["id"])
        if cached and trainable(cached):
            joined.append(
                dict(
                    row,
                    **{
                        k: cached[k]
                        for k in (
                            "cache",
                            "key",
                            "num_frames",
                            "video_shape",
                            "config_hash",
                        )
                    },
                )
            )
    rows = joined
    after = Counter(x for row in rows for x in row["l2_ids"])
    cats = {x["id"]: x for x in stats["categories"]["l2"]}
    targets = r["categories"] or [
        x["id"]
        for x in stats["categories"]["l2"]
        if x["pool_count"] >= r["min_category_size"]
    ]
    if r["mode"] == "mixed":
        targets = [x for x in targets if r["mix"].get(x, r["mix"]["default"]) > 0]
    if not targets:
        raise ValueError("没有参与训练的非零权重 L2，请调整类别或门槛")
    missing = [x for x in targets if not after[x]]
    if missing:
        raise ValueError("过滤后以下 L2 无数据: " + ", ".join(missing))
    if r["mode"] == "per_category":
        tail = [x for x in targets if after[x] < r["min_category_size"]]
        if tail:
            raise ValueError("过滤后以下 L2 不足最小条数: " + ", ".join(tail))
    catalog = store.read(DATA / "catalog.json", [])
    plans = []
    for subset in [targets] if r["mode"] == "mixed" else [[x] for x in targets]:
        candidate = [row for row in rows if set(row["l2_ids"]) & set(subset)]
        before_mix = len(candidate)
        weighted = (
            _weighted(candidate, subset, r["mix"], r["target_size"], r["train"]["seed"])
            if r["mode"] == "mixed"
            else candidate[:]
        )
        unique = {row["id"]: row for row in weighted}
        heldout = cases.select(
            list(unique.values()),
            subset,
            1 if r["mode"] == "mixed" else r["eval"]["heldout_per_category"],
            r["eval"]["max_cases"],
            r["train"]["seed"],
        )
        covered = {x for c in heldout for x in c["covered_l2"]}
        uncovered = set(subset) - covered
        groups = {c["source"] for c in heldout}
        training = [row for row in weighted if row["source_group"] not in groups]
        remaining = Counter(x for row in training for x in row["l2_ids"])
        local_warnings = warnings[:]
        if r["mode"] == "mixed" and uncovered:
            raise ValueError(
                "测试未覆盖以下 L2，或留出会清空其训练来源；请排除这些类别或增大 max_cases: "
                + ", ".join(sorted(uncovered))
            )
        if any(not remaining[x] for x in subset):
            raise ValueError(
                "留出后以下 L2 无训练数据，请排除: "
                + ", ".join(x for x in subset if not remaining[x])
            )
        if (
            r["mode"] == "per_category"
            and len(heldout) < r["eval"]["heldout_per_category"]
        ):
            local_warnings.append(f"{subset[0]} 仅选出 {len(heldout)} 条独立留出样本")
        if not heldout:
            raise ValueError("无法在保留训练来源的同时选择留出样本")
        products = cases.products(
            catalog,
            subset,
            after,
            r["eval"]["products"],
            r["train"]["seed"],
            r["mode"] == "mixed",
        )
        if len(products) < r["eval"]["products"]:
            local_warnings.append(
                f"产品动作不足，实际生成 {len(products)} 条产品描述用例"
            )
        selected = heldout + products
        for i, c in enumerate(selected):
            c["seed"] = r["train"]["seed"] + i
        category = "mixed" if r["mode"] == "mixed" else subset[0]
        name = (
            ("混合·" + (r["name"] or batch.name))
            if category == "mixed"
            else cats[category]["name"]
        )
        result = RunPlan(
            category,
            name,
            subset,
            training,
            len(candidate),
            selected,
            groups,
            before,
            after,
            before_mix,
            len(weighted),
            r,
            local_warnings,
            dict(
                downsampled=len(candidate) - len(unique),
                heldout_rows=len(weighted) - len(training),
            ),
        )
        if (
            len(training) < len(r["train"]["gpus"]) * r["train"]["batch"]
            or not result.preview()["epoch_steps"]
        ):
            raise ValueError(name + " 的同形状完整 batch 不足，不能启动训练")
        plans.append(result)
    return plans


def materialize(recipe, parse_id=None, dispatch=False):
    if not isinstance(recipe, dict):
        raise ValueError("配方须为 JSON 对象")
    batch = store.batch_path(recipe.get("dataset"))
    with store.lock(batch):
        plans = _plan(recipe, batch)
        r = plans[0].recipe
        # Validate heldout caches too: evaluation reads them unconditionally.
        index = load_index_by_id()
        rng = random.Random(r["train"]["seed"])
        checked = {}
        for p in plans:
            unique = {row["id"]: row for row in p.rows}
            sampled = rng.sample(list(unique.values()), min(200, len(unique)))
            bad = []
            for row in sampled:
                if row["id"] not in checked:
                    checked[row["id"]] = validate_cache(row)
                if checked[row["id"]]:
                    bad.append(dict(id=row["id"], error=checked[row["id"]]))
            if len(bad) / len(sampled) > 0.01:
                raise ValueError(
                    f"缓存抽检失败 {len(bad)}/{len(sampled)}，失败率超过 1%：{bad[:3]}"
                )
        failures = [
            dict(id=rid, error=error) for rid, error in checked.items() if error
        ]
        if failures:
            plans = _plan(recipe, batch, invalid_ids={f["id"] for f in failures})
            for p in plans:
                p.warnings.append(
                    f"已排除抽检发现的 {len(failures)} 条坏缓存，并重新计算留出与配比"
                )
        for p in plans:
            for c in p.cases:
                if c["kind"] == "留出样本":
                    error = validate_cache(dict(index[c["name"]], caption=c["prompt"]))
                    if error:
                        raise ValueError("留出缓存无效 " + c["name"] + "：" + error)
        summary = admission(batch)
        recipe_id = (
            store.now()[:19].replace("-", "").replace(":", "").replace("T", "-")
            + "-"
            + secrets.token_hex(3)
        )
        run_ids = []
        created = []
        csv_rows = []
        RUNS.mkdir(exist_ok=True)
        try:
            for p in plans:
                from datetime import datetime
                from zoneinfo import ZoneInfo

                run_id = (
                    datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
                    + "-"
                    + secrets.token_hex(3)
                )
                run = RUNS / run_id
                run.mkdir()
                created.append(run)
                cfg = dict(
                    r["train"],
                    category=p.category,
                    name=p.name,
                    products=sum(c["kind"] == "产品描述" for c in p.cases),
                    limit=0,
                    sampling_steps=r["eval"]["sampling_steps"],
                    prepared_data=True,
                    dataset=batch.name,
                    recipe_id=recipe_id,
                    mode=r["mode"],
                    dispatch_authorized=dispatch,
                    annotation_source=f"datasets/{batch.name}/qc/summary.json",
                )
                store.write(run / "config.json", cfg)
                # Serialize original index rows, including exact repetitions.
                store.write_jsonl(
                    run / "train.jsonl", (index[row["id"]] for row in p.rows)
                )
                store.write(run / "cases.json", p.cases)
                l2counts = Counter(x for row in p.rows for x in row["l2_ids"])
                store.write(
                    run / "split.json",
                    dict(
                        matched=p.matched,
                        train=len(p.rows),
                        heldout=sum(c["kind"] == "留出样本" for c in p.cases),
                        groups_excluded=len(p.heldout_groups),
                        l2_counts=dict(l2counts),
                        sources=dict(
                            Counter(
                                row.get("dataset_source") or "null" for row in p.rows
                            )
                        ),
                    ),
                )
                store.write_jsonl(
                    run / "test_prompts.jsonl", cases.manifest(p.cases, p.targets)
                )
                source = store.read(batch / "source.json")
                stats = store.read(batch / "stats.json")
                catnames = {c["id"]: c["name"] for c in stats["categories"]["l2"]}
                unique_counts = Counter(
                    x
                    for row in {row["id"]: row for row in p.rows}.values()
                    for x in row["l2_ids"]
                )
                # Include co-occurring labels: weight changes also amplify unselected labels.
                all_l2 = sorted(set(p.before_filter) | set(l2counts))
                mix = dict(
                    source={k: source[k] for k in ("path", "sha256")},
                    dataset=batch.name,
                    qc=dict(
                        pass_rate=summary["pass_rate"],
                        gate=summary["gate"],
                        excluded=len(summary["excluded_ids"]),
                        sample_sha256=summary["sample_sha256"],
                        reviews_sha256=summary["reviews_sha256"],
                    ),
                    recipe_id=recipe_id,
                    mode=r["mode"],
                    filters=r["filters"],
                    weights=r["mix"] if r["mode"] == "mixed" else {"default": 1},
                    per_l2=[
                        dict(
                            id=x,
                            name=catnames.get(x, x),
                            pool=p.before_filter[x],
                            after_filter=p.after_filter[x],
                            weight=r["mix"].get(x, r["mix"]["default"])
                            if r["mode"] == "mixed" and x in p.targets
                            else 1
                            if r["mode"] == "per_category" and x in p.targets
                            else 0,
                            rows_in_train=l2counts[x],
                            ratio_in_train=l2counts[x] / len(p.rows),
                            label_occurrences=l2counts[x],
                            unique_rows=unique_counts[x],
                        )
                        for x in all_l2
                    ],
                    totals=dict(
                        pool=stats["trainability"]["pool"],
                        unique_rows=len({row["id"] for row in p.rows}),
                        rows_after_mix=p.after_mix,
                        train_rows=len(p.rows),
                        duplicated=len(p.rows) - len({row["id"] for row in p.rows}),
                        downsampled=p.mix_counts["downsampled"],
                        heldout_rows=p.mix_counts["heldout_rows"],
                        heldout_groups=len(p.heldout_groups),
                    ),
                    train_sha256=sha256_file(run / "train.jsonl"),
                )
                store.write(run / "mix.json", mix)
                store.write(run / "recipe.json", r)
                store.write(
                    run / "validation.json",
                    dict(
                        cache_samples=len(checked),
                        failures=failures,
                        preview=p.preview(),
                    ),
                )
                store.write(run / "state.json", dict(stage="数据已核验"))
                run_ids.append(run_id)
                csv_rows.append(
                    dict(
                        run_id=run_id,
                        mode=r["mode"],
                        category=p.category,
                        train_rows=len(p.rows),
                        unique_rows=mix["totals"]["unique_rows"],
                        heldout=sum(c["kind"] == "留出样本" for c in p.cases),
                        lr=r["train"]["lr"],
                        batch=r["train"]["batch"],
                        steps=r["train"]["steps"],
                        gpus=",".join(map(str, r["train"]["gpus"])),
                        recipe_id=recipe_id,
                        created_at=store.now(),
                    )
                )
            # Final admission check immediately precedes committing the recipe record.
            admission(batch)
            parsed = (
                store.read(batch / "recipes" / (parse_id + ".json"))
                if parse_id
                else None
            )
            record = dict(
                recipe_id=recipe_id,
                recipe=r,
                runs=run_ids,
                created_at=store.now(),
                prompt=(parsed or {}).get("prompt"),
                llm_raw=(parsed or {}).get("llm_raw"),
                parse_id=parse_id,
                preview=[p.preview() for p in plans],
            )
            store.write(batch / "recipes" / (recipe_id + ".json"), record)
            path = batch / "runs.csv"
            previous = path.read_text() if path.exists() else ""
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(csv_rows[0]))
            if not previous:
                writer.writeheader()
            writer.writerows(csv_rows)
            with store.atomic_stream(path) as stream:
                stream.write(previous + buf.getvalue())
        except Exception:
            for run in created:
                shutil.rmtree(run)
            raise
        return run_ids
