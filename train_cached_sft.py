"""Full SFT on D_human_train native 480p shape buckets."""

import argparse
import gzip
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel
from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("WAN_DATA_DIR", ROOT / "studio/data")).expanduser()
INDEX = os.environ.get("WAN_INDEX", str(DATA / "index.jsonl.gz"))
MODEL = os.environ.get("WAN_BASE_MODEL", str(ROOT / "outputs/l2-sport-fullbody/base"))


class CachedVideos(torch.utils.data.Dataset):
    def __init__(self, index, frames, portrait, limit):
        self.rows = []
        self.buckets = defaultdict(list)
        h, w = (104, 60) if portrait else (60, 104)
        self.shape = (16, (frames + 3) // 4, h, w) if frames else None
        self.root = Path(index).parent
        opener = gzip.open if str(index).endswith(".gz") else open
        with opener(index, "rt", encoding="utf-8") as manifest:
            for line in manifest:
                if not line.strip():
                    continue
                row = json.loads(line)
                shape = tuple(row["video_shape"])
                c, t, height, width = shape
                if (
                    c == 16
                    and 12 <= t <= 25
                    and sorted((height, width)) == [60, 104]
                    and row["num_frames"] == 4 * t - 3
                    and (self.shape is None or self.shape == shape)
                ):
                    self.buckets[shape].append(len(self.rows))
                    self.rows.append((row["cache"], row["key"], row["config_hash"], shape))
                    if limit and len(self.rows) == limit:
                        break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, key, config_hash, shape = self.rows[idx]
        record = torch.load(self.root / path, weights_only=True, mmap=True, map_location="cpu")
        video, text = record["video_latent"], record["text_embedding"]
        if (
            record["key"] != key
            or record["config_hash"] != config_hash
            or record["metadata"]["fps"] != 24
            or tuple(video.shape) != shape
            or text.ndim != 2
            or text.shape[1] != 4096
            or not 0 < len(text) <= 512
        ):
            raise ValueError(f"Invalid cache entry: {path}")
        return video, torch.nn.functional.pad(text, (0, 0, 0, 512 - len(text)))


class ShapeBatches:
    """Shuffle full local batches and group similar frame counts into each DDP update."""

    def __init__(self, buckets, batch, rank=0, world=1, seed=42):
        self.rank, self.world, self.seed = rank, world, seed
        self.epoch = 0
        self.buckets = buckets
        self.batch = batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        lengths = []
        for shape, indices in self.buckets.items():
            perm = torch.randperm(len(indices), generator=generator).tolist()
            for offset in range(0, len(indices) - self.batch + 1, self.batch):
                batches.append([indices[i] for i in perm[offset : offset + self.batch]])
                lengths.append(shape[1])
        order = torch.randperm(len(batches), generator=generator).tolist()
        if self.world == 1:
            yield from (batches[i] for i in order)
            return

        # Select the same complete batches as the shuffled sampler before regrouping.
        # All supported 480p orientations have equal spatial token counts.
        steps = len(self)
        selected = order[: steps * self.world]
        selected.sort(key=lengths.__getitem__)
        for step in torch.randperm(steps, generator=generator).tolist():
            yield batches[selected[step * self.world + self.rank]]

    def __len__(self):
        return sum(len(indices) // self.batch for indices in self.buckets.values()) // self.world


def load_model(folder):
    config = json.loads((folder / "config.json").read_text())
    if config["model_type"] != "t2v" or config["dim"] != 1536:
        raise ValueError("Expected a native Wan2.1-T2V-1.3B checkpoint")
    state = convert_wan_transformer_to_diffusers(
        load_file(str(folder / "diffusion_pytorch_model.safetensors"))
    )
    with torch.device("meta"):
        model = WanTransformer3DModel(
            num_attention_heads=config["num_heads"],
            attention_head_dim=config["dim"] // config["num_heads"],
            num_layers=config["num_layers"],
            ffn_dim=config["ffn_dim"],
            freq_dim=config["freq_dim"],
            eps=config["eps"],
        )
    model.load_state_dict(state, strict=True, assign=True)
    # RoPE has unpersisted buffers. Construct those on CPU before moving the module.
    model.rope = type(model.rope)(config["dim"] // config["num_heads"], (1, 2, 2), 1024)
    model.enable_gradient_checkpointing()
    return model.to(device="cuda", dtype=torch.float32).train()


def run(a):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl")
    torch.manual_seed(a.seed + rank)
    dataset = CachedVideos(a.index, a.frames, a.portrait, a.limit)
    if len(dataset) < a.batch:
        raise ValueError(f"Only {len(dataset)} matching records for batch {a.batch}")
    if a.frames and world == 1:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True
        )
    else:
        batches = ShapeBatches(dataset.buckets, a.batch, rank, world, a.seed)
        if not len(batches):
            raise ValueError("No complete batch with matching geometry")
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batches, num_workers=a.workers, pin_memory=True
        )
    if a.save and Path(a.save).exists():
        raise FileExistsError(a.save)
    model = load_model(Path(a.model))
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], gradient_as_bucket_view=True
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), fused=True)
    iterator = iter(loader)
    durations = []
    steps = len(loader) if a.one_epoch else a.warmup + a.steps
    if rank == 0:
        print(
            f"records={len(dataset)} steps={steps} world={world} global_batch={world * a.batch} lr={a.lr}",
            flush=True,
        )
    for step in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            video, text = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            video, text = next(iterator)
        video = video.cuda(non_blocking=True).float()
        text = text.cuda(non_blocking=True)
        noise = torch.randn_like(video)
        u = torch.rand(a.batch, device="cuda")
        sigma = 3 * u / (1 + 2 * u)
        noisy = torch.lerp(video, noise, sigma[:, None, None, None, None])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(noisy, sigma * 1000, text, return_dict=False)[0]
            loss = torch.nn.functional.mse_loss(pred.float(), noise - video)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not math.isfinite(loss.item()):
            raise ValueError("Nonfinite loss")
        if step >= a.warmup:
            durations.append(elapsed)
        if rank == 0:
            print(
                f"step={step + 1} seconds={elapsed:.3f} loss={loss.item():.5f} grad_norm={norm.item():.4f}",
                flush=True,
            )
    result = dict(
        status="ok",
        gpu=torch.cuda.get_device_name(),
        frames=a.frames,
        shape=dataset.shape,
        batch=a.batch,
        warmup=a.warmup,
        world_size=world,
        seconds=durations,
        samples_s=world * a.batch * len(durations) / sum(durations),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        allocator=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")),
        num_records=len(dataset),
        workers=a.workers,
        learning_rate=a.lr,
        torch_version=str(torch.__version__),
        seed=a.seed,
        index=a.index,
        model=a.model,
    )
    if rank == 0 and a.save:
        destination = Path(a.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            staging = Path(tmp) / "transformer"
            transformer = model.module if world > 1 else model
            transformer.save_pretrained(staging)
            os.rename(staging, destination)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default=INDEX)
    p.add_argument("--model", default=MODEL)
    p.add_argument(
        "--frames",
        type=int,
        default=97,
        choices=[0, *range(45, 98, 4)],
        help="0 mixes all native lengths and orientations",
    )
    p.add_argument("--portrait", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--limit", type=int, default=128, help="First N matching records; 0 means entire bucket")
    p.add_argument("--one-epoch", action="store_true", help="One pass through complete distributed batches")
    p.add_argument("--steps", type=int, default=3, help="Timed optimizer updates after warmup")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save", help="Optional final Diffusers transformer directory")
    args = p.parse_args()
    if min(args.batch, args.steps) < 1 or min(args.warmup, args.workers, args.limit) < 0:
        p.error("Invalid batch, steps, limit, workers or warmup")
    if args.one_epoch and args.warmup:
        p.error("--one-epoch requires --warmup 0")
    if args.frames == 0 and args.portrait:
        p.error("--portrait requires a fixed frame count")
    try:
        report = run(args)
    except torch.cuda.OutOfMemoryError as exc:
        print(
            json.dumps(dict(status="oom", batch=args.batch, frames=args.frames, error=str(exc))), flush=True
        )
        raise SystemExit(42)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report), flush=True)
