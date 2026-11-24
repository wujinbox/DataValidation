# DataValidation

面向视频数据的批次统计、抽样质检、训练配方和 Wan2.1 缓存微调工作台。

## 下载与启动

```bash
git clone --recurse-submodules https://github.com/wujinbox/DataValidation.git
cd DataValidation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python studio/server.py
```

使用 Linux 和 Python 3.10 或更新版本。浏览器打开 http://127.0.0.1:28765 。在远端启动时，通过 SSH 端口转发在本机打开。已有检出可以用 `git submodule update --init --recursive` 补齐子模块。

仓库固定了 [FastVideo](https://github.com/hao-ai-lab/FastVideo/tree/866a3cdb91a85aef8561847365fc8564add9e4c8) 和 [Wan2.2](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff) 的源码版本。FastVideo 可按其 README 单独安装。工作台的 SFT 使用 PyTorch 和 Diffusers，评测使用 Wan2.2 的 VAE 和 T5 模块。

## 用合成数据试跑质检

安装带 `libx264` 的 FFmpeg。在已激活的虚拟环境中执行以下命令，然后打开「数据批次」和「数据质检」。

```bash
demo_dir=$(mktemp -d)
python studio/demo_data.py "$demo_dir/data"
export WAN_INDEX="$demo_dir/data/index.jsonl"
batch=$(cd studio && python -m datapipe.cli profile "$demo_dir/data/demo.jsonl" |
  python -c 'import json,sys; print(json.load(sys.stdin)["batch_id"])')
(cd studio && python -m datapipe.cli sample "$batch")
python studio/server.py
```

合成视频可用于统计图、播放、抽样、人工审核和准入流程。示例索引只含供质检抽样使用的元数据，训练需要另外准备真实张量缓存。

## 接入实际数据和 GPU

1. UI 登记视频清单的绝对路径。每行是 JSON 对象，需要唯一 `id`，并提供可读的 `video_path`、`caption`、`width`、`height`、`fps`、`duration` 和 `l2_category_ids`。可参考 `studio/demo_data.py` 中生成的清单。
2. 设置 `WAN_INDEX` 为匹配视频 ID 的 JSONL 或 JSONL.GZ 缓存索引。训练条目需有 `id`、`cache`、`key`、`config_hash`、`num_frames`、`video_shape` 和 `sampling.l2_ids`。为工作台提供计算节点和控制端共同可读的绝对缓存路径。每份 PyTorch 缓存含对应的 `key`、`config_hash`、`caption`、`metadata.fps`（24）、`video_latent` 和 `text_embedding`；尺寸约束见 `studio/common.py`。
3. 在计算节点准备适配 CUDA 的 PyTorch 环境，并安装 `requirements-gpu.txt` 中的依赖。节点需要 `nvidia-smi` 和带 `drawtext`、`libx264` 的 FFmpeg。基座目录使用原版 Wan2.1-T2V-1.3B 的 `config.json` 和 `diffusion_pytorch_model.safetensors`。评测需要 `Wan2.1_VAE.pth`；包含产品描述用例时还需要 `models_t5_umt5-xxl-enc-bf16.pth` 和 `google/umt5-xxl/`。负向提示词缓存为 PyTorch 文本嵌入张量 `negative.pt`。模型权重和实际缓存由部署者提供。
4. 控制端和计算节点以相同绝对路径访问批次、缓存和任务目录。控制端通过 SSH 的非交互认证访问选定 GPU 节点。启动工作台前配置下表中的路径。单独跑训练时可以用 `python train_cached_sft.py --index /path/to/index.jsonl --model /path/to/Wan2.1-T2V-1.3B --frames 0 --limit 0 --steps 100 --save /path/to/new-output`。

权重可从 [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) 和 [Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) 获取。已有负向嵌入可以直接接入。如需重新生成，先配置 `WAN_ENCODER_DIR`，再在 GPU 节点运行：

```bash
python studio/evaluate.py negative "blur, artifacts, flicker" --output /path/to/negative.pt
```

| 环境变量 | 用途 |
| --- | --- |
| `WAN_DATA_DIR` | 可选数据目录，默认 `studio/data`。可提供 `catalog.json` 以加载类目详情和产品动作描述 |
| `WAN_INDEX` | 缓存索引路径，默认 `$WAN_DATA_DIR/index.jsonl.gz`（工作台） |
| `WAN_STUDIO_DATA_PYTHON` | 控制端数据处理解释器，默认工作台所用的 Python |
| `WAN_STUDIO_HOSTS` | GPU 节点编号到 SSH 目标的 JSON 对象，如 `{"0008":"user@gpu.example"}`。默认无远端节点 |
| `WAN_STUDIO_REMOTE_PYTHON` | 节点上的训练环境 Python 绝对路径 |
| `WAN_STUDIO_REMOTE_DIR` | 节点上的 `studio` 目录绝对路径 |
| `WAN_STUDIO_REMOTE_OUTPUT` | 为已有 `local_output` 任务指定节点输出根目录 |
| `WAN_BASE_MODEL` | Wan2.1-T2V-1.3B 基座目录 |
| `WAN_ENCODER_DIR` | 含 VAE 和 T5 权重及 tokenizer 的目录 |
| `WAN_NEGATIVE_EMBEDDING` | 负向提示词嵌入文件绝对路径 |
| `WAN_WAN_MODULES` | 可选，覆盖默认 Wan2.2 子模块的 `wan/modules` 路径 |
| `WAN_STUDIO_PORT` | 工作台监听端口，默认 28765 |
| `WAN_FEISHU_SPACE`、`WAN_FEISHU_PARENT` | 启用飞书归档时同时设置知识空间 ID 和父节点 token，并在控制端授权 `lark-cli` |
| `WAN_FEISHU_REFERENCE` | 可选，归档文档里的类目参考链接 |

基础工作流使用视频 JSONL 中的类目标签。若需要原有产品动作描述用例，可提供 `catalog.json`，或用 `python studio/catalog.py` 从数据目录中的 `index-config.json`、`parents.json`、`sheet.json`、`products.json` 与缓存索引生成。可选 `WAN_ACTION_ALIASES` 指向含 `l3_to_l2` 映射的 JSON。部署时使用自己的类目、飞书节点和表格导出。

更多操作见 [工作台使用说明](studio/README.md) 和 [缓存 SFT 说明](CACHED_SFT.md)。

## 验证

```bash
cd studio
python -m unittest discover -s datapipe/tests -v
```

浏览器测试另需 Playwright、Chromium 和 FFmpeg，运行 `python datapipe/tests/browser_smoke.py`。
