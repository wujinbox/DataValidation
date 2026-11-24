"""Build the L2 catalogue from the curated manifest and the product team's sheet."""

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("WAN_DATA_DIR", ROOT / "data")).expanduser().resolve()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    # Unique temporary names also make independent writers safe.
    import os
    import tempfile

    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def records():
    path = Path(os.environ.get("WAN_INDEX", DATA / "index.jsonl.gz")).expanduser()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def norm(s):
    return re.sub(r"\s+", "", s or "").replace("（", "(").replace("）", ")")


def build():
    taxonomy = read(DATA / "index-config.json")["taxonomy"]
    parents = {
        norm(n): v["l2_id"] for n, v in read(DATA / "parents.json")["mapping"].items()
    }
    legacy_aliases = ROOT.parent.parent / (
        "D_human_train/extensions/source_l2_sampling/configs/legacy_action_aliases.json"
    )
    aliases_path = os.environ.get("WAN_ACTION_ALIASES")
    aliases = {}
    if aliases_path or legacy_aliases.is_file():
        aliases = read(Path(aliases_path).expanduser() if aliases_path else legacy_aliases)["l3_to_l2"]
    by_name = {v: k for k, v in taxonomy.items()}
    for action, parent_name in aliases.items():
        if parent_name in by_name:
            parents.setdefault(norm(action), by_name[parent_name])
    counts = Counter()
    for row in records():
        counts.update(set(row["sampling"]["l2_ids"]))
    cats = {
        k: dict(id=k, name=v, count=counts[k], actions=[])
        for k, v in sorted(taxonomy.items())
    }
    rows = read(DATA / "sheet.json")["sheets"][0]["data"]
    products = read(DATA / "products.json")["sheets"][0]["data"]
    assert len(rows) == len(products) and products[0][0] == "产品动作描述"
    sheet_names = {norm(r[0]) for r in rows[1:] if r[0]}
    for n, (r, p) in enumerate(zip(rows[1:], products[1:]), 2):
        if r[0]:
            parent = r[0].strip()
        if not r[4] or not p[0]:
            continue
        name = r[4].strip()
        direct = next((k for k, v in taxonomy.items() if norm(v) == norm(parent)), None)
        mapped = parents.get(norm(name))
        keys = {direct} if direct else set()
        if mapped in cats and norm(taxonomy[mapped]) not in sheet_names:
            keys.add(mapped)
        if not keys:
            continue
        full = (
            p[3] == "是"
            or str(r[8]).strip() in ("全身", "腿", "脚", "下肢", "膝")
            or parent in ("站立脚步", "行走类", "转身类", "蹲起类", "脚部细节", "运动")
        )
        frame = "全身" if full else "半身"
        description = str(p[0]).strip()
        scene = (
            "A realistic video of a single adult woman with shoulder-length dark hair, "
            "wearing a plain light-gray short-sleeved T-shirt, dark straight-leg trousers, "
            "and white sneakers. She is centered in a spacious room with off-white walls "
            "and a light wooden floor. Soft, even daylight illuminates her face and clothing. "
            "The background is uncluttered and remains stationary."
        )
        camera = (
            "A locked-off wide shot shows her entire body from head to toe, both feet, "
            "and the surrounding floor, with ample space above and beside her for movement."
            if full
            else "A locked-off medium shot frames her from the waist up, keeping her head, "
            "upper body, arms, and hands visible with space for gestures."
        )
        for key in keys:
            cats[key]["actions"].append(
                dict(
                    sheet_l2=parent,
                    name=name,
                    row=n,
                    code=r[3],
                    framing=frame,
                    definition=r[13],
                    description=description,
                    prompt=scene + " " + camera + " " + description,
                )
            )
    write(DATA / "catalog.json", list(cats.values()))


if __name__ == "__main__":
    build()
