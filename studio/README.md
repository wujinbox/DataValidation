# Wan 动作实验台

默认服务绑定回环端口 28765。安装和数据接入步骤见[仓库首页](../README.md)。若程序在远端运行，在浏览器所在机器建立转发后打开 http://127.0.0.1:28765 。

```bash
ssh -N -L 28765:127.0.0.1:28765 user@studio-host
```

「训练」页面统一提供训练设置、提交和任务进度。选择已通过质检的数据批次，再选择类别、数据配比与训练参数，预览后提交；任务及历史记录显示在同一页面下方。可刷新计算资源查看 GPU 状态并选卡。学习率默认 3e-6，步数 0 表示按原始尺寸分桶训练一轮完整分布式 batch。关闭浏览器不影响已经提交的任务。

每次运行在同目录的 runs 下生成独立目录。config.json 记录参数，train.jsonl 是排除留出来源后的清单，cases.json 记录全部测试 prompt、来源或产品表行、镜头和种子，job.log 记录训练步数、耗时、损失与评测进度。训练结束后保存 transformer 权重，用同一噪声生成每个用例的基座和 SFT 视频。如果配置了飞书归档，工作台会在指定父节点下生成子文档。归档失败可从实验记录重试，已完成的评测会复用。

如果通过 systemd 部署，可用服务的实际名称查询状态和日志。

```bash
systemctl --user status wan-l2-console
journalctl --user -u wan-l2-console -n 30
```

## 数据批次 → 质检 → 训练配方

工作台提供「训练」「数据批次」「数据质检」三个标签页。「训练配方」和「训练任务」已合并为同一「训练」页面，不再单列旧的快速启动入口；已有任务记录仍保留。

1. 在「数据批次」登记可读的 JSONL 绝对路径。后台生成统计、中文报告、七张图和默认 20 条抽样；相同内容的文件返回已有批次。
2. 在「数据质检」填写审核人，逐条播放视频、审核 caption 和 L2 标签。点击标签可标记错误，缺失标签通过下拉列表追加。`P` / `F` 保存并跳转，左右箭头切换；在文字输入框内不会触发快捷键。
3. 全部审完后点击「完成质检」。默认通过率门槛 90%，未达到时可填写放行理由，否则保持「待处理」。不通过的样本始终排除。追加类别抽检或修改审核会使原准入结果失效，需要再次完成质检。
4. 在「训练」页面的「新建训练任务」中编辑全部参数，或先用自然语言生成配方，再修改表单。点击预览检查训练行、测试覆盖、步数和时间估算；修改表单后必须重新预览。最后确认提交。提交后设置区自动收起，并定位到下方任务记录；可重新展开设置区继续新建任务。
5. `per_category` 为每个所选 L2 建立独立任务；`mixed` 生成一个混合任务。同一配方的任务依次执行，所选 GPU 在队列结束前预留。训练任务中可以停止排队中的单个任务。

每个 run 保存 `mix.json`、`test_prompts.jsonl` 和 `recipe.json` 等产物，可从训练记录直接打开。配比按多标签最大权重取整重复／下采样实现；测试来源整组从训练集排除。报告记录实际配比，多标签占比合计可能超过 100%。

最短边、景别、方向和来源过滤在构建训练集时应用；基础训练池只检查缓存命中、去重和训练器尺寸兼容性。缺缓存条目保存在 `uncached.jsonl`，本功能不负责补缓存。`target_size` 在剔除留出来源之前应用，所以最终训练行数以预览为准。每类都必须保有训练数据，混合任务必须覆盖全部参与类别的留出测试。

抽检缓存的失败率超过 1% 时构建失败；不超过 1% 时剔除已发现的坏缓存，重新计算留出和配比。测试用例所用缓存会额外校验。完整 batch 按实际形状分桶计算，而非简单用总行数除以全局 batch。

服务重启后恢复队列顺序，不重复启动已有监督进程。若进程意外退出且无法确认执行状态，任务显示「恢复待确认」并继续预留资源；先停止该任务，再重新提交。单独使用 CLI `build` 只落盘，不加入自动执行队列；通过页面提交才授权排队。已排队任务在启动前也会重新核对质检结果。

## 配置

服务端和数据处理子进程继承启动服务时的环境变量：

| 环境变量 | 说明 |
| --- | --- |
| `WAN_INDEX` | 可覆盖 `studio/data/index.jsonl.gz`，支持 JSONL 或 gzip JSONL；索引更新后进程内缓存自动失效 |
| `WAN_QC_THRESHOLD` | 质检通过率门槛，0–1，默认 0.9 |
| `WAN_RECIPE_LLM` | `anthropic`（默认）或 `openai` |
| `WAN_RECIPE_MODEL` | Anthropic 默认 `claude-opus-5-5`，OpenAI 默认 `gpt-4.1`；可配置账户支持的模型 |
| `WAN_RECIPE_BASE_URL` | 可选服务地址；OpenAI 填含 `/v1` 的根地址，Anthropic 支持根地址或 `/v1` 地址 |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | 对应提供方的密钥；没有密钥时自然语言接口返回中文提示，表单仍可使用 |

密钥不要写入配方或代码。LLM 仅接收用户的配方文本、类别目录及类别数量，不接收 caption、视频路径或人物 IP。解析超时为 60 秒，非法配方最多修正重试一次。实现使用标准库 HTTP，不要求 server 安装 SDK；OpenAI 使用 [Chat Completions 接口](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)，Anthropic 使用 [Messages 接口](https://platform.claude.com/docs/en/api/messages)。

批次产物保存在 `studio/datasets/<batch_id>/`；审核记录只追加，最后一条生效。配方记录保留自然语言输入、模型原文、确认后的配方与展开的任务。JSON、JSONL、报告、图表使用临时文件加原子替换；追加审核记录在批次锁下写入并刷新到磁盘。服务仍只监听 127.0.0.1，视频仅能访问抽样清单中的文件，支持拖动播放。

## 命令行与验证

从 studio 目录运行：

```bash
PY=python
$PY -m datapipe.cli profile /absolute/path/batch.jsonl
$PY -m datapipe.cli sample '<batch_id>' --n 20 --seed 42
$PY -m datapipe.cli sample '<batch_id>' --extra-l2 l1-05-l2-01 --k 3
$PY -m datapipe.cli finish '<batch_id>'
$PY -m datapipe.cli plan '<batch_id>' --recipe /absolute/path/recipe.json
$PY -m datapipe.cli build '<batch_id>' --recipe /absolute/path/recipe.json
python3 -m unittest discover -s datapipe/tests -v
```

`plan` 不写 run；`build` 校验后落盘但不启动任务。二者均要求真实批次通过质检准入。已经生成质检样本的批次不能重新统计覆盖，避免破坏审核关联。源文件或索引版本改变时构建会拒绝使用过期统计。

单元测试只使用 `/tmp` 合成样本和训练／发布替身。可选浏览器验收脚本为 `datapipe/tests/browser_smoke.py`，需要 Playwright 和 Chromium；它使用临时服务，验证三个标签页和统一训练入口、视频拖动、审核持久化、准入、无密钥回退、配方预览和提交流程，不会启动 GPU 或向飞书发布。
