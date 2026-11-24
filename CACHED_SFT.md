# Wan2.1-T2V-1.3B 单卡缓存 SFT

在能读取训练索引、缓存文件和原版模型目录的 GPU 节点运行。先按[仓库首页](README.md)准备 CUDA Python 环境。

```bash
GPU=0 PYTORCH_ALLOC_CONF=expandable_segments:True bash bench_cached_sft.sh \
  --index /path/to/index.jsonl --model /path/to/Wan2.1-T2V-1.3B
```

每个 batch 用单独的进程从同一权重开始。每档做一次预热更新、三次计时更新，直到实际 CUDA OOM。计时包含取缓存、传输、前向、反向、梯度裁剪和 AdamW 更新。默认测 97 和 77 帧横屏，使用每桶前 128 条记录打乱取样。设 `FRAMES=97` 可只测一个长度，传 `--portrait` 可测竖屏。每档逐步耗时、loss、梯度范数与显存保存在新建的 bench 目录。

训练使用 Diffusers Wan DiT，严格匹配并加载本地原版 1.3B 权重。缓存中的 Wan2.1 VAE latent 已在编码时归一化，直接用于 flow matching；文本取缓存的 UMT5 embedding。输入仅含视频和文本。FP32 主权重和 AdamW 状态，BF16 autocast，整块梯度检查点，梯度累积为 1。

下例读取完整索引，按原始 latent 的 T/H/W 分桶，每桶不足一个本地 batch 的余数在该轮跳过。多卡先随机选出能组成完整分布式步的本地 batch，再按帧数相近分配到同一步，最后随机打乱步序。横竖屏可以分配到同一步的不同卡。每个 batch 都是独立的真实样本。可用 `--index`、`--model` 和 `--workers` 覆盖数据、权重和加载进程数。

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python train_cached_sft.py \
  --index /path/to/index.jsonl --model /path/to/Wan2.1-T2V-1.3B \
  --frames 0 --limit 0 --batch 4 --steps 100 --warmup 0 \
  --save outputs/sft-100/transformer
```

`--save` 在最后一步导出 Diffusers transformer 权重；输出目录须为新目录。测速默认只写日志。混合训练按桶内样本均匀打乱；原项目的音频、头部参考条件及 source/L2 加权采样均不参与这次 T2V 训练。
