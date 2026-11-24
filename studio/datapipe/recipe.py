"""Recipe validation and a privacy-bounded LLM adapter (standard library only)."""

import copy
import json
import math
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from cluster import HOSTS
from . import store

DEFAULTS = dict(
    mode="mixed",
    categories=[],
    mix={"default": 1.0},
    target_size=0,
    filters=dict(
        min_short_side=480,
        body_framing=["half_body", "close_up", "three_quarter_body", "full_body"],
        orientation=["landscape", "portrait"],
        exclude_review_required=False,
        exclude_body_framing_mixed=False,
        max_per_ip=0,
        dataset_source=[],
    ),
    min_category_size=200,
    eval=dict(heldout_per_category=5, max_cases=80, products=2, sampling_steps=40),
    train=dict(host="0008", gpus=[0, 1, 2, 3], lr=3e-6, batch=2, steps=0, seed=42),
    name="",
)


def integer(v, label, lo, hi):
    if type(v) is not int or not lo <= v <= hi:
        raise ValueError(f"{label} 须为 {lo}–{hi} 的整数")
    return v


def numeric(v, label, lo, hi, positive=False):
    if (
        type(v) not in (int, float)
        or not math.isfinite(v)
        or not lo <= v <= hi
        or (positive and v == 0)
    ):
        raise ValueError(
            f"{label} 须在 " + ("(0, " if positive else f"[{lo}, ") + f"{hi}] 内"
        )


def strings(v, label, allowed=None, empty=True):
    if (
        not isinstance(v, list)
        or any(not isinstance(x, str) for x in v)
        or (not empty and not v)
    ):
        raise ValueError(label + " 须为字符串列表")
    if allowed is not None and any(x not in allowed for x in v):
        raise ValueError(
            label + " 含不存在的选项: " + ", ".join(x for x in v if x not in allowed)
        )
    return list(dict.fromkeys(v))


def validate(recipe, stats):
    if not isinstance(stats, dict) or "categories" not in stats:
        raise ValueError("统计尚未完成，请稍后再配置训练")
    if not isinstance(recipe, dict):
        raise ValueError("配方须为 JSON 对象")
    allowed = set(DEFAULTS) | {"dataset"}
    if set(recipe) - allowed:
        raise ValueError("配方含未知字段: " + ", ".join(sorted(set(recipe) - allowed)))
    r = copy.deepcopy(DEFAULTS)
    for key, value in recipe.items():
        if key in ("filters", "eval", "train"):
            if not isinstance(value, dict) or set(value) - set(DEFAULTS[key]):
                raise ValueError(key + " 包含无效字段")
            r[key].update(value)
        else:
            r[key] = copy.deepcopy(value)
    dataset = r.get("dataset")
    if (
        not isinstance(dataset, str)
        or not store.ID.fullmatch(dataset)
        or dataset in (".", "..")
    ):
        raise ValueError("请指定有效 dataset 批次 ID")
    if r["mode"] not in ("per_category", "mixed"):
        raise ValueError("训练模式须为 per_category 或 mixed")
    cats = {c["id"]: c for c in stats["categories"]["l2"]}
    r["categories"] = strings(
        r["categories"], "categories", cats, empty=r["mode"] == "mixed"
    )
    minimum = integer(r["min_category_size"], "类别最小条数", 1, 10000000)
    if r["mode"] == "per_category":
        tail = [x for x in r["categories"] if cats[x]["pool_count"] < minimum]
        if tail:
            raise ValueError(
                "以下长尾 L2 不足最小条数，无法单独训练: " + ", ".join(tail)
            )
    if not isinstance(r["mix"], dict) or any(
        k != "default" and k not in cats for k in r["mix"]
    ):
        raise ValueError("mix 含不存在的 L2")
    r["mix"].setdefault("default", 1.0)
    for k, v in r["mix"].items():
        numeric(v, "配比 " + k, 0, 10)
    integer(r["target_size"], "目标行数", 0, 10000000)
    f = r["filters"]
    integer(f["min_short_side"], "最短边", 0, 100000)
    f["body_framing"] = strings(
        f["body_framing"],
        "景别",
        {"half_body", "close_up", "three_quarter_body", "full_body", "unknown"},
        empty=False,
    )
    f["orientation"] = strings(
        f["orientation"], "画幅", {"landscape", "portrait", "square"}, empty=False
    )
    f["dataset_source"] = strings(f["dataset_source"], "dataset_source")
    integer(f["max_per_ip"], "每个 IP 最大条数", 0, 10000000)
    for key in ("exclude_review_required", "exclude_body_framing_mixed"):
        if type(f[key]) is not bool:
            raise ValueError(key + " 须为布尔值")
    e = r["eval"]
    for key, lo, hi in [
        ("heldout_per_category", 1, 100),
        ("max_cases", 1, 1000),
        ("products", 0, 1000),
        ("sampling_steps", 2, 100),
    ]:
        integer(e[key], key, lo, hi)
    t = r["train"]
    cards = t["gpus"]
    if not isinstance(t["host"], str) or t["host"] not in HOSTS:
        raise ValueError("训练节点不存在")
    if (
        not isinstance(cards, list)
        or not 1 <= len(cards) <= 8
        or any(type(x) is not int or not 0 <= x <= 7 for x in cards)
        or len(set(cards)) != len(cards)
    ):
        raise ValueError("请选择同一节点上不重复的 0–7 号 GPU")
    t["gpus"] = sorted(cards)
    numeric(t["lr"], "学习率", 0, 1e-3, positive=True)
    integer(t["batch"], "每卡 batch", 1, 6)
    integer(t["steps"], "训练步数", 0, 100000)
    integer(t["seed"], "seed", 0, 2147483647)
    if not isinstance(r["name"], str) or len(r["name"]) > 200:
        raise ValueError("配方名称最多 200 字")
    warnings = []
    if r["mode"] == "per_category" and (
        r["target_size"] or r["mix"] != {"default": 1.0}
    ):
        warnings.append("单类别模式忽略配比及目标行数")
    return r, warnings


def extract(text):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
    raise ValueError("模型未输出有效 JSON 对象")


def _request(provider, model, system, messages, timeout):
    key = os.environ.get(
        "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    )
    base = os.environ.get("WAN_RECIPE_BASE_URL") or (
        "https://api.anthropic.com"
        if provider == "anthropic"
        else "https://api.openai.com/v1"
    )
    base = base.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if provider == "anthropic":
        url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        body = dict(model=model, max_tokens=4096, system=system, messages=messages)
    else:
        url = base + "/chat/completions"
        headers["Authorization"] = "Bearer " + key
        body = dict(
            model=model, messages=[dict(role="system", content=system)] + messages
        )
    try:
        with urlopen(
            Request(
                url, json.dumps(body, ensure_ascii=False).encode(), headers=headers
            ),
            timeout=timeout,
        ) as response:
            data = json.load(response)
    except HTTPError as exc:
        # Do not echo provider bodies, URLs or credentials into browser errors.
        raise ValueError(
            f"配方服务返回 HTTP {exc.code}，请检查模型和服务配置"
        ) from None
    except (URLError, TimeoutError) as exc:
        raise ValueError("配方服务连接失败或超时，请使用表单") from exc
    if provider == "anthropic":
        return "".join(
            x.get("text", "") for x in data["content"] if x.get("type") == "text"
        )
    return data["choices"][0]["message"]["content"]


def parse(prompt, batch_dir):
    from .build import plan

    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise ValueError("请输入 1–12000 字的训练需求")
    provider = os.environ.get("WAN_RECIPE_LLM", "anthropic")
    if provider not in ("anthropic", "openai"):
        raise ValueError("WAN_RECIPE_LLM 须为 anthropic 或 openai")
    if not os.environ.get(
        "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    ):
        raise ValueError("未配置配方服务密钥，请使用表单")
    stats = store.read(batch_dir / "stats.json")
    catalog = [
        {k: c[k] for k in ("id", "name", "l1", "pool_count", "long_tail")}
        for c in stats["categories"]["l2"]
    ]
    defaults = dict(dataset=batch_dir.name, **DEFAULTS)
    # Only public taxonomy and counts are included; never serialize stats/source/pool.
    system = (
        "将用户需求转换为训练配方，只输出 JSON 对象，不要解释。默认配方及全部字段如下："
        + json.dumps(defaults, ensure_ascii=False)
        + "\n字段约束：mode 为 per_category 或 mixed；categories 为目录内 L2 ID，单类别模式必填，mixed 空列表表示全部非长尾类别；mix 为 default 和 L2 权重，范围 [0,10]，取多标签最大权重；target_size=0 不缩放；"
        "filters 的列表选择要保留的值，dataset_source=[] 不限来源，max_per_ip=0 不限。单类最小条数默认200。"
        "eval.heldout_per_category=1..100，max_cases=1..1000，products=0..1000，sampling_steps=2..100。"
        "train.host 为四位节点号，gpus 为同一节点不重复的0..7，lr 在(0,0.001]，batch=1..6，steps=0..100000（0表示一个epoch），seed=0..2147483647。"
        "目录为：" + json.dumps(catalog, ensure_ascii=False)
    )
    model = os.environ.get(
        "WAN_RECIPE_MODEL", "claude-opus-5-5" if provider == "anthropic" else "gpt-4.1"
    )
    messages = [dict(role="user", content=prompt)]
    raw = []
    started = time.monotonic()
    for attempt in range(2):
        remaining = 55 - (time.monotonic() - started)
        if remaining <= 0:
            raise ValueError("配方服务超时，请使用表单")
        reply = _request(provider, model, system, messages, remaining)
        raw.append(reply)
        try:
            r = extract(reply)
            r["dataset"] = batch_dir.name
            r, warnings = validate(r, stats)
            plans = plan(r)
            return dict(
                recipe=r,
                warnings=warnings + [w for p in plans for w in p.warnings],
                preview=[p.preview() for p in plans],
                llm_raw=raw,
                prompt=prompt,
                provider=provider,
                model=model,
            )
        except ValueError as exc:
            if attempt:
                raise ValueError("模型配方校验失败，请使用表单：" + str(exc))
            messages.extend(
                [
                    dict(role="assistant", content=reply),
                    dict(role="user", content="请修正配方，错误为：" + str(exc)),
                ]
            )
