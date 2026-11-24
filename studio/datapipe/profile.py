"""One-pass tolerant JSONL profiling. Plotting is imported only in this process."""

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import json
import math
import time

from catalog import DATA
from common import load_index_by_id, index_version, source_group, trainable
from . import store

KNOWN = set(
    "id video_path duration start end fps width height screen_orientation source_screen_orientation is_square body_framing body_framing_mixed body_framing_review_recommended l1_category l1_category_ids l2_category l2_category_ids l3_category l3_category_source category_schema category_status category_prediction_id global_ip_id global_id ip_status ip_threshold ip_cluster_space caption static_caption dynamic_caption gesture_caption review_required language_review dataset_source dataset_split pipeline_version".split()
)


def number(v):
    return type(v) in (int, float) and math.isfinite(v)


def quantiles(values):
    if not values:
        return dict(p5=None, p50=None, p95=None, min=None, max=None)
    a = sorted(values)

    def q(p):
        pos = (len(a) - 1) * p
        low = int(pos)
        return a[low] + (a[min(low + 1, len(a) - 1)] - a[low]) * (pos - low)

    return dict(p5=q(0.05), p50=q(0.5), p95=q(0.95), min=a[0], max=a[-1])


def histogram(values, bins=12):
    if not values:
        return dict(edges=[], counts=[])
    low, high = min(values), max(values)
    if low == high:
        high = low + 1
    step = (high - low) / bins
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, int((value - low) / step))] += 1
    return dict(edges=[low + i * step for i in range(bins + 1)], counts=counts)


def gini(counts):
    a = sorted(counts)
    total = sum(a)
    return (
        (
            2 * sum((i + 1) * x for i, x in enumerate(a)) / (len(a) * total)
            - (len(a) + 1) / len(a)
        )
        if total
        else 0
    )


def profile(batch_dir):
    batch = Path(batch_dir)
    with store.lock(batch):
        if (batch / "qc/sample.json").exists():
            raise ValueError("已有质检样本，禁止重新统计覆盖；请登记新的数据版本")
        try:
            return _profile(batch)
        except Exception as exc:
            store.state(batch, "失败", error=str(exc))
            raise


def _profile(batch):
    started = time.monotonic()
    src = store.read(batch / "source.json")
    path = Path(src["path"])
    st = path.stat()
    if st.st_size != src["size"] or st.st_mtime_ns != src["mtime_ns"]:
        raise ValueError("源文件已变化，请重新登记")
    store.state(batch, "统计中", progress=0)
    index = load_index_by_id()
    version = index_version()
    catalog = store.read(DATA / "catalog.json", [])
    names = {c["id"]: c["name"] for c in catalog}
    discovered_names = {}
    warnings = set()
    fields, badtypes, nulls, drops = Counter(), Counter(), Counter(), Counter()
    wh, short, aspects, orientations, framing, fps = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    l1, l2, labels, cooccur, language, ip = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    l2ip, l2frame, l2orient = (
        defaultdict(Counter),
        defaultdict(Counter),
        defaultdict(Counter),
    )
    pools = Counter()
    sources = {
        k: Counter()
        for k in (
            "dataset_source",
            "dataset_split",
            "pipeline_version",
            "category_schema",
        )
    }
    numeric = defaultdict(list)
    auto = {}
    seen = set()
    anomalies = []
    mixed = recommended = reviews = mismatch = hit = shape_ok = frames_ok = (
        pool_size
    ) = 0
    invalid = duplicates = rows = valid = 0

    def string(row, key):
        value = row.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            badtypes[key] += 1
            return None
        return value

    def flag(row, key):
        value = row.get(key)
        if value is not None and type(value) is not bool:
            badtypes[key] += 1
        return value is True

    def items(row, key):
        value = row.get(key)
        if value is None:
            return []
        if not isinstance(value, list):
            badtypes[key] += 1
            return []
        if any(not isinstance(x, str) for x in value):
            badtypes[key] += 1
        return [x for x in value if isinstance(x, str) and x]

    with (
        store.atomic_stream(batch / "pool.jsonl") as pool_file,
        store.atomic_stream(batch / "uncached.jsonl") as miss_file,
        path.open(encoding="utf-8", errors="replace") as stream,
    ):
        for line in stream:
            rows += 1
            if rows % 10000 == 0:
                store.state(batch, "统计中", progress=rows)
            try:
                r = json.loads(line)
                if not isinstance(r, dict):
                    raise ValueError()
            except (ValueError, TypeError):
                invalid += 1
                drops["invalid_json"] += 1
                continue
            valid += 1
            fields.update(r.keys())
            for k, v in r.items():
                if v is None:
                    nulls[k] += 1
                if k in KNOWN or v is None:
                    continue
                entry = auto.setdefault(
                    k, dict(types=Counter(), dist=Counter(), numbers=[], capped=False)
                )
                kind = (
                    "categorical"
                    if isinstance(v, (str, bool))
                    else "numeric"
                    if number(v)
                    else "list"
                    if isinstance(v, list)
                    else "unsupported"
                )
                entry["types"][kind] += 1
                if kind == "numeric":
                    entry["numbers"].append(v)
                elif kind in ("categorical", "list"):
                    vals = v if kind == "list" else [v]
                    for val in vals:
                        if not isinstance(val, (str, bool, int, float)):
                            continue
                        val = str(val)
                        if val in entry["dist"] or len(entry["dist"]) < 200:
                            entry["dist"][val] += 1
                        else:
                            entry["capped"] = True
            rid = string(r, "id")
            for key in ("duration", "start", "end", "width", "height", "fps"):
                value = r.get(key)
                if value is not None and not number(value):
                    badtypes[key] += 1
            width = r.get("width") if number(r.get("width")) else None
            height = r.get("height") if number(r.get("height")) else None
            orient = string(r, "screen_orientation")
            frame = string(r, "body_framing")
            wh[f"{width}x{height}"] += 1
            orientations[orient or "null"] += 1
            framing[frame or "null"] += 1
            if width and height and width > 0 and height > 0:
                side = min(width, height)
                short[
                    "<480"
                    if side < 480
                    else "480-719"
                    if side < 720
                    else "720-1079"
                    if side < 1080
                    else ">=1080"
                ] += 1
                ratio = width / height
                aspect = next(
                    (
                        name
                        for name, value in [
                            ("16:9", 16 / 9),
                            ("9:16", 9 / 16),
                            ("4:3", 4 / 3),
                            ("1:1", 1),
                        ]
                        if abs(ratio / value - 1) < 0.02
                    ),
                    "other",
                )
                aspects[aspect] += 1
                computed = (
                    "landscape"
                    if width > height
                    else "portrait"
                    if width < height
                    else "square"
                )
                mismatch += bool(orient and orient != computed)
                if (width % 2 or height % 2 or width % 16 or height % 16) and len(
                    anomalies
                ) < 100:
                    anomalies.append(
                        dict(
                            id=rid,
                            reason=f"非16对齐尺寸 width={width}, height={height}",
                        )
                    )
            else:
                short["null"] += 1
                aspects["null"] += 1
            duration = r.get("duration")
            if number(duration) and duration >= 0:
                numeric["duration"].append(duration)
                numeric["frames"].append(round(duration * 24))
            if number(r.get("fps")):
                fps[str(round(r["fps"], 3))] += 1
            else:
                fps["null"] += 1
            mixed += flag(r, "body_framing_mixed")
            recommended += flag(r, "body_framing_review_recommended")
            review = flag(r, "review_required")
            reviews += review
            for key in ("static_caption", "dynamic_caption", "caption"):
                text = string(r, key)
                if text is not None:
                    numeric[key].append(len(text.split()))
            lr = r.get("language_review")
            if isinstance(lr, list):
                for item in lr:
                    if isinstance(item, dict) and isinstance(item.get("code"), str):
                        language[item["code"]] += 1
            elif lr is not None:
                badtypes["language_review"] += 1
            l1ids, l2ids = items(r, "l1_category_ids"), items(r, "l2_category_ids")
            for prefix, ids in [("l1", l1ids), ("l2", l2ids)]:
                ns = items(r, prefix + "_category")
                if len(ids) != len(ns):
                    warnings.add(prefix + " 标签 ID 与名称数量不一致")
                for cid, name in zip(ids, ns):
                    if cid in names and names[cid] != name:
                        warnings.add(f"{cid} 名称与 catalog 不一致，以 jsonl 为准")
                    if cid in discovered_names and discovered_names[cid] != name:
                        warnings.add(f"{cid} 在 jsonl 中存在多个名称，以最后出现的为准")
                    discovered_names[cid] = name
            l1.update(set(l1ids))
            l2.update(set(l2ids))
            l2ids = sorted(set(l2ids))
            labels[str(len(l2ids))] += 1
            cooccur.update(combinations(l2ids, 2))
            person = string(r, "global_ip_id") or string(r, "global_id")
            if person:
                ip[person] += 1
            for cid in l2ids:
                if person:
                    l2ip[cid][person] += 1
                l2frame[cid][frame or "null"] += 1
                l2orient[cid][orient or "null"] += 1
            for k, counter in sources.items():
                counter[string(r, k) or "null"] += 1
            cached = index.get(rid)
            if cached:
                hit += 1
                shape = cached.get("video_shape", [])
                shape_ok += bool(
                    len(shape) == 4
                    and shape[0] == 16
                    and sorted(shape[2:]) == [60, 104]
                )
                frames_ok += bool(
                    type(cached.get("num_frames")) is int
                    and 45 <= cached["num_frames"] <= 97
                )
            else:
                miss_file.write(
                    json.dumps(
                        dict(id=rid, line=rows, reason="cache_miss"), ensure_ascii=False
                    )
                    + "\n"
                )
            duplicate = rid in seen if rid else False
            if rid:
                seen.add(rid)
            duplicates += duplicate
            reason = (
                "missing_id"
                if not rid
                else "duplicate_id"
                if duplicate
                else "cache_miss"
                if not cached
                else "incompatible_cache"
                if not trainable(cached)
                else None
            )
            if reason:
                drops[reason] += 1
                continue
            item = dict(
                id=rid,
                l2_ids=l2ids,
                l1_ids=l1ids,
                l1_names=items(r, "l1_category"),
                l2_names=[discovered_names.get(k, names.get(k, k)) for k in l2ids],
                body_framing=frame,
                orientation=orient,
                ip=person,
                source_group=source_group(r),
                duration=duration if number(duration) else None,
                width=width,
                height=height,
                fps=r.get("fps") if number(r.get("fps")) else None,
                review_required=review,
                body_framing_mixed=r.get("body_framing_mixed") is True,
                dataset_source=r.get("dataset_source")
                if isinstance(r.get("dataset_source"), str)
                else None,
                video_path=string(r, "video_path"),
                caption=string(r, "caption") or "",
                static_caption=string(r, "static_caption"),
                dynamic_caption=string(r, "dynamic_caption"),
                language_review=lr if isinstance(lr, list) else [],
            )
            pool_file.write(
                json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
            )
            pools.update(l2ids)
            pool_size += 1
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (src["size"], src["mtime_ns"]):
        raise ValueError("源文件在统计期间发生变化，请重新登记")
    if index_version() != version:
        raise ValueError("缓存索引在统计期间发生变化，请重试")
    names.update(discovered_names)
    auto_fields = {}
    for k, v in auto.items():
        kind = v["types"].most_common(1)[0][0]
        auto_fields[k] = dict(
            type=kind,
            dist=quantiles(v["numbers"]) if kind == "numeric" else dict(v["dist"]),
            types=dict(v["types"]),
            truncated=v["capped"],
        )
    category_rows = [
        dict(
            id=k,
            name=names.get(k, k),
            l1=k.split("-l2-")[0],
            count=v,
            ratio=v / max(valid, 1),
            ip_count=len(l2ip[k]),
            top_ip_share=max(l2ip[k].values(), default=0) / max(v, 1),
            pool_count=pools[k],
            long_tail=pools[k] < 200,
        )
        for k, v in sorted(l2.items())
    ]
    stats = dict(
        meta=dict(
            source=src["path"],
            sha256=src["sha256"],
            rows=rows,
            valid_rows=valid,
            invalid_rows=invalid,
            unique_ids=len(seen),
            dup_ids=duplicates,
            schema_fields=dict(fields),
            missing_fields={k: valid - fields[k] for k in sorted(KNOWN | set(fields))},
            null_fields=dict(nulls),
            invalid_types=dict(badtypes),
            unknown_fields=sorted(set(fields) - KNOWN),
            warnings=sorted(warnings),
            generated_at=store.now(),
            index_version=list(version),
        ),
        resolution=dict(
            wh_top=wh.most_common(50),
            short_side_bins={
                k: short[k] for k in ("<480", "480-719", "720-1079", ">=1080", "null")
            },
            aspect=dict(aspects),
            orientation=dict(orientations),
            orientation_mismatch=mismatch,
            anomalies=anomalies,
        ),
        duration=dict(
            hist=histogram(numeric["duration"]), **quantiles(numeric["duration"])
        ),
        fps=dict(values=dict(fps)),
        frames_24fps=dict(
            hist=histogram(numeric["frames"]),
            lt45=sum(x < 45 for x in numeric["frames"]),
            **{"45-97": sum(45 <= x <= 97 for x in numeric["frames"])},
            gt97=sum(x > 97 for x in numeric["frames"]),
        ),
        framing=dict(
            body_framing=dict(framing), mixed=mixed, review_recommended=recommended
        ),
        categories=dict(
            l1=[
                dict(id=k, name=names.get(k, k), count=v, ratio=v / max(valid, 1))
                for k, v in sorted(l1.items())
            ],
            l2=category_rows,
            labels_per_sample=dict(labels),
            l2_x_framing=dict(l2frame),
            l2_x_orientation=dict(l2orient),
            l2_cooccurrence_top=[[a, b, n] for (a, b), n in cooccur.most_common(50)],
        ),
        ip=dict(
            unique=len(ip),
            per_ip_hist={
                k: sum(1 for n in ip.values() if low <= n <= high)
                for k, low, high in [
                    ("1", 1, 1),
                    ("2-5", 2, 5),
                    ("6-20", 6, 20),
                    ("21-100", 21, 100),
                    (">100", 101, float("inf")),
                ]
            },
            top20=ip.most_common(20),
            top20_share=sum(n for _, n in ip.most_common(20)) / max(valid, 1),
            gini=gini(ip.values()),
        ),
        text=dict(
            static_words=quantiles(numeric["static_caption"]),
            dynamic_words=quantiles(numeric["dynamic_caption"]),
            caption_words=quantiles(numeric["caption"]),
            review_required=reviews,
            language_review_codes=dict(language),
        ),
        source={k: dict(v) for k, v in sources.items()},
        auto_fields=auto_fields,
        trainability=dict(
            cache_hit=hit,
            cache_miss=valid - hit,
            shape_ok=shape_ok,
            frames_ok=frames_ok,
            pool=pool_size,
            drop_reasons=dict(drops),
        ),
    )
    src["rows"] = rows
    store.write(batch / "source.json", src)
    store.write(batch / "stats.json", stats)
    charts(batch, stats, ip)
    report(batch, stats)
    store.write(
        batch / "profile-complete.json",
        dict(sha256=src["sha256"], index_version=list(version)),
    )
    elapsed = time.monotonic() - started
    store.state(batch, "待质检", progress=rows, seconds=round(elapsed, 3))
    return stats


def charts(batch, s, ips):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wan-studio-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = batch / "charts"
    folder.mkdir(exist_ok=True)

    def save(name, labels, counts, horizontal=False, colors=None):
        fig, ax = plt.subplots(
            figsize=(10, max(4, len(labels) * 0.19)) if horizontal else (8, 4)
        )
        if horizontal:
            ax.barh(labels, counts, color=colors or "#287454")
            ax.invert_yaxis()
        else:
            ax.bar(range(len(labels)), counts, color=colors or "#287454")
            ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_title(name)
        fig.tight_layout()
        with store.atomic_stream(folder / (name + ".png"), binary=True) as output:
            fig.savefig(output, format="png", dpi=110)
        plt.close(fig)

    h = s["duration"]["hist"]
    edges = h["edges"]
    save(
        "duration_hist",
        [f"{a:.1f}-{b:.1f}" for a, b in zip(edges, edges[1:])],
        h["counts"],
    )
    for name, dist in [
        ("short_side_bins", s["resolution"]["short_side_bins"]),
        ("framing", s["framing"]["body_framing"]),
        ("labels_per_sample", s["categories"]["labels_per_sample"]),
    ]:
        save(name, list(dist), list(dist.values()))
    cats = s["categories"]
    save("l1_bar", [x["id"] for x in cats["l1"]], [x["count"] for x in cats["l1"]])
    l2 = sorted(cats["l2"], key=lambda x: -x["count"])
    save(
        "l2_bar",
        [x["id"] for x in l2],
        [x["count"] for x in l2],
        True,
        ["#aaa" if x["long_tail"] else "#287454" for x in l2],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    a = sorted(ips.values())
    total = sum(a)
    cumulative = 0
    xs, ys = [0], [0]
    for i, n in enumerate(a):
        cumulative += n
        xs.append((i + 1) / len(a))
        ys.append(cumulative / total)
    ax.plot(xs, ys)
    ax.plot([0, 1], [0, 1], "--", color="#aaa")
    ax.set_title("ip_lorenz")
    ax.set_xlabel("Fraction of IPs")
    ax.set_ylabel("Fraction of samples")
    fig.tight_layout()
    with store.atomic_stream(folder / "ip_lorenz.png", binary=True) as output:
        fig.savefig(output, format="png", dpi=110)
    plt.close(fig)


def report(batch, s):
    def clean(v):
        return str(v).replace("|", "／").replace("\n", " ").replace("\r", " ")

    def table(headers, rows):
        return (
            "\n".join(
                [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                ]
                + ["| " + " | ".join(clean(v) for v in row) + " |" for row in rows]
            )
            + "\n"
        )

    def dist(d):
        return table(["项目", "条数"], d.items())

    def img(name):
        return f"\n![{name}](charts/{name}.png)\n"

    def pct(x):
        return f"{100 * x:.2f}%"

    m = s["meta"]
    n = max(m["valid_rows"], 1)
    t = s["trainability"]
    c = s["categories"]
    r = s["resolution"]
    sections = [
        f"# 视频数据分布报告\n\n## 1. 数据概览\n\n来源：{m['source']}\n\nSHA256：{m['sha256']}\n\n总行数 {m['rows']}，有效行 {m['valid_rows']}，坏行 {m['invalid_rows']}，唯一 ID {m['unique_ids']}，重复 ID {m['dup_ids']}。训练池 {t['pool']} 条（{pct(t['pool'] / n)}），缓存命中率 {pct(t['cache_hit'] / n)}。\n\n"
        + dist(t["drop_reasons"]),
        "## 2. 分辨率与画幅\n\n"
        + dist(r["short_side_bins"])
        + "\n"
        + table(["分辨率", "条数"], r["wh_top"][:10])
        + "\n"
        + dist(r["orientation"])
        + f"\n方向标记不一致：{r['orientation_mismatch']} 条。\n\n"
        + table(["ID", "异常"], [(a["id"], a["reason"]) for a in r["anomalies"][:20]])
        + img("short_side_bins"),
        "## 3. 时长、帧率与可训练帧\n\n"
        + table(
            ["指标", "P5", "P50", "P95", "最小", "最大"],
            [
                ["时长（秒）"]
                + [s["duration"][k] for k in ("p5", "p50", "p95", "min", "max")]
            ],
        )
        + "\n"
        + dist(s["fps"]["values"])
        + "\n"
        + dist({k: v for k, v in s["frames_24fps"].items() if k != "hist"})
        + "\n24fps 帧数按原视频时长估算，实际训练使用缓存帧数。\n"
        + img("duration_hist"),
        "## 4. 景别\n\n"
        + dist(s["framing"]["body_framing"])
        + f"\n混合景别 {s['framing']['mixed']}，建议复核 {s['framing']['review_recommended']}。\n\n"
        + table(
            ["L2", "景别分布"],
            [
                (x["id"], json.dumps(c["l2_x_framing"][x["id"]], ensure_ascii=False))
                for x in sorted(c["l2"], key=lambda x: -x["count"])[:30]
            ],
        )
        + img("framing"),
        "## 5. 类别\n\n"
        + table(
            ["L1", "名称", "条数", "占比"],
            [(x["id"], x["name"], x["count"], pct(x["ratio"])) for x in c["l1"]],
        )
        + "\n"
        + table(
            [
                "L2",
                "名称",
                "L1",
                "条数",
                "占比",
                "池内",
                "IP 数",
                "头部 IP 占比",
                "长尾",
            ],
            [
                (
                    x["id"],
                    x["name"],
                    x["l1"],
                    x["count"],
                    pct(x["ratio"]),
                    x["pool_count"],
                    x["ip_count"],
                    pct(x["top_ip_share"]),
                    "是" if x["long_tail"] else "否",
                )
                for x in c["l2"]
            ],
        )
        + "\n"
        + dist(c["labels_per_sample"])
        + "\n"
        + table(["L2 A", "L2 B", "共现数"], c["l2_cooccurrence_top"])
        + img("l1_bar")
        + img("l2_bar")
        + img("labels_per_sample"),
        f"## 6. 人物 IP 集中度\n\nIP 数 {s['ip']['unique']}，Top-20 占比 {pct(s['ip']['top20_share'])}，Gini {s['ip']['gini']:.4f}。\n\n"
        + table(["IP", "条数"], s["ip"]["top20"])
        + "\n"
        + "".join(
            f"- {x['id']} 头部 IP 占比 {pct(x['top_ip_share'])}，存在集中风险。\n"
            for x in c["l2"]
            if x["top_ip_share"] > 0.3
        )
        + img("ip_lorenz"),
        "## 7. 文本质量\n\n"
        + table(
            ["文本", "P5 词数", "P50", "P95"],
            [
                [k] + [s["text"][k][q] for q in ("p5", "p50", "p95")]
                for k in ("caption_words", "static_words", "dynamic_words")
            ],
        )
        + f"\nreview_required {s['text']['review_required']} 条（{pct(s['text']['review_required'] / n)}）。\n\n"
        + dist(s["text"]["language_review_codes"]),
        "## 8. 来源与版本\n\n"
        + "\n".join("### " + k + "\n\n" + dist(v) for k, v in s["source"].items()),
        "## 9. 自动发现的其他字段\n\n"
        + table(
            ["字段", "类型", "分布／分位数"],
            [
                (k, v["type"], json.dumps(v["dist"], ensure_ascii=False))
                for k, v in s["auto_fields"].items()
            ],
        )
        + "\n### 缺字段统计\n\n"
        + dist({k: v for k, v in m["missing_fields"].items() if v})
        + "\n### 类型异常\n\n"
        + dist(m["invalid_types"]),
        "## 10. 结论与建议\n\n"
        + f"- {sum(x['long_tail'] for x in c['l2'])} 个 L2 的池内条数低于默认门槛 200，不建议单独训练。\n"
        + f"- 竖屏占比 {pct(r['orientation'].get('portrait', 0) / n)}，需关注画幅覆盖。\n"
        + f"- 短边 <480 占比 {pct(r['short_side_bins'].get('<480', 0) / n)}，配方默认排除；基础训练池尚未应用此过滤。\n"
        + "- 仅抽 20 条时统计效力有限：通过率 90% 的 95% 区间近似 ±13%，可追加类别抽检。\n"
        + "".join("- " + w + "\n" for w in m["warnings"]),
    ]
    with store.atomic_stream(batch / "分布报告.md") as stream:
        stream.write("\n\n".join(sections) + "\n")
