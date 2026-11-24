"""Publish one recoverable child report per experiment."""

from html import escape
import json
import os
from pathlib import Path
from statistics import mean
import shutil
import time
import subprocess
from xml.etree import ElementTree as ET

from catalog import read, write


def lark(run, *args):
    proc = subprocess.run(
        [
            shutil.which("lark-cli") or str(Path.home() / ".local/bin/lark-cli"),
            *args,
            "--as",
            "user",
        ],
        cwd=run,
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-1800:] or proc.stdout[-1800:])
    result = json.loads(proc.stdout)
    if not result["ok"]:
        raise RuntimeError(str(result.get("error")))
    return result["data"]


def data_tables(run, cases):
    """Pure local rendering, independently testable without accessing Feishu."""
    if not (run / "mix.json").exists():
        return ""
    mix = read(run / "mix.json")
    rows = []
    for row in mix.get("per_l2", []):
        if row["rows_in_train"]:
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{escape(str(v))}</td>"
                    for v in (
                        row["id"],
                        row["name"],
                        row["rows_in_train"],
                        f"{row['ratio_in_train']:.2%}",
                    )
                )
                + "</tr>"
            )
    html = (
        "<h2>数据配比</h2><p>多标签样本允许重叠，占比合计可能超过 100%。</p><table><tr><th>L2</th><th>名称</th><th>训练行</th><th>实际占比</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    coverage = {}
    for case in cases:
        for cid in case.get("covered_l2", []):
            coverage.setdefault(cid, []).append(case["name"])
    html += "<h2>L2 → 测试样例</h2><table><tr><th>L2</th><th>样例</th></tr>"
    html += "".join(
        f"<tr><td>{escape(cid)}</td><td>{escape('、'.join(names))}</td></tr>"
        for cid, names in sorted(coverage.items())
    )
    return html + "</table>"


def publish(run):
    parent = os.environ["WAN_FEISHU_PARENT"]
    space = os.environ["WAN_FEISHU_SPACE"]
    sheet = os.environ.get("WAN_FEISHU_REFERENCE")
    cfg = read(run / "config.json")
    cases = read(run / "cases.json")
    split = read(run / "split.json")
    title = f"Wan 1.3B · {cfg['name']} · {run.name}"
    if (run / "node.json").exists():
        node = read(run / "node.json")
    else:

        def lookup():
            siblings = lark(
                run,
                "wiki",
                "+node-list",
                "--space-id",
                space,
                "--parent-node-token",
                parent,
                "--page-all",
                "--page-limit",
                "0",
            )
            return next((n for n in siblings["nodes"] if n["title"] == title), None)

        node = lookup()
        if node is None:
            if not (run / "created.json").exists():
                created = lark(
                    run, "docs", "+create", "--parent-token", parent, "--title", title
                )
                write(run / "created.json", created)
            for delay in (0, 1, 2, 4, 8):
                time.sleep(delay)
                node = lookup()
                if node:
                    break
            if not node:
                raise RuntimeError("已提交文档创建，暂未在指定父节点查到，可重试归档")
        write(run / "node.json", node)
    obj = node["obj_token"]

    def content():
        return lark(run, "docs", "+fetch", "--doc", obj, "--detail", "full")[
            "document"
        ]["content"]

    def append(xml):
        (run / "upload.xml").write_text(xml)
        lark(
            run,
            "docs",
            "+update",
            "--doc",
            obj,
            "--command",
            "append",
            "--doc-format",
            "xml",
            "--content",
            "@./upload.xml",
        )

    heldout = [i for i, c in enumerate(cases) if "cache" in c]
    summary = (
        f"<h1>训练与评测结果</h1><p>{escape(cfg['name'])} · 480p T2V · {split['matched']} 条匹配 · "
        f"{split['train']} 条进入训练清单。留出测试 {len(heldout)} 条，产品描述测试 {len(cases) - len(heldout)} 条。"
        "每对共享 prompt、seed、初始噪声和采样器。左为基座，右为 SFT。</p>"
    )
    if heldout:
        before, after = [
            mean(read(run / f"{i:02}-{v}.json")["flow_mse"] for i in heldout)
            for v in ("base", "sft")
        ]
        summary += (
            f"<p>留出样本固定噪声、t=500 的平均 flow MSE，基座 {before:.5f}，SFT {after:.5f}。"
            "该指标反映去噪误差，动作自然度和指令遵循需结合视频判断。</p>"
        )
    if cfg.get("dataset"):
        mix = read(run / "mix.json") if (run / "mix.json").exists() else {}
        qc = mix.get("qc", {})
        summary += (
            f"<p>数据批次 {escape(cfg['dataset'])}，抽样质检通过率 {qc.get('pass_rate', 0):.1%}，"
            f"准入状态 {escape(qc.get('gate', '未知'))}；已排除 {qc.get('excluded', 0)} 条审核不通过样本。"
            "质检为抽样审核，未对全部训练样本逐条审核。</p>"
        )
    elif cfg.get("annotation_source"):
        summary += f"<p>人工审核通过数据来源 {escape(cfg['annotation_source'])}。{split['train']} 条均标注 caption 准确、动作自然。本次从原始 Wan2.1-T2V-1.3B 基座微调。</p>"
    summary += (
        f"<p>学习率 {cfg['lr']}，每卡 batch {cfg['batch']}，GPU {escape(str(cfg['gpus']))}，"
        f"训练预算 {cfg['steps'] or '1 epoch'}，UniPC {cfg['sampling_steps']} 步，CFG 5，shift 3，24 fps。</p>"
    )
    if sheet:
        summary += f'<p><a href="{escape(sheet, quote=True)}">动作定义与产品描述</a></p>'
    summary += data_tables(run, cases)
    current = content()
    if "训练与评测结果" not in current:
        append(summary)
    existing = {
        s.get("name")
        for s in ET.fromstring("<root>" + current + "</root>").iter("source")
        if s.get("token")
    }
    for i, c in enumerate(cases):
        filename = f"pair-{i:02}.mp4"
        if filename in existing:
            continue
        field = (
            f"表格第 {c['row']} 行 · {c['framing']} · {escape(str(c['definition']))}"
            if "row" in c
            else f"验证来源 {escape(c['source'])} · {'L2' if cfg.get('dataset') else 'L3'} {escape(', '.join(c['labels']) or '未标注')}"
        )
        append(
            f"<h2>{i + 1}. {c['kind']} · {escape(c['name'])}</h2><p>{field} · seed {c['seed']}</p>"
            f'<p>{escape(c["prompt"])}</p><figure view-type="Preview"><source path="@./{filename}" name="{filename}"/></figure>'
        )
    verified = ET.fromstring("<root>" + content() + "</root>")
    attached = {s.get("name") for s in verified.iter("source") if s.get("token")}
    if not {f"pair-{i:02}.mp4" for i in range(len(cases))} <= attached:
        raise RuntimeError("飞书附件回读未通过，请检查后重试发布")
    return "https://ecnoie7eb92s.feishu.cn/wiki/" + node["node_token"]
