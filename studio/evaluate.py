"""Paired T2V sampling and matched held-out flow loss, on one selected GPU."""

import gc
import importlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import torch
from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
from catalog import ROOT, read, write

sys.path.insert(0, str(ROOT.parent))
from train_cached_sft import load_model

BASE = ROOT.parent / "outputs/l2-sport-fullbody/base"
ENC = Path("/mnt/nvme0n1/models/Wan2.2-S2V-14B")
DTYPE = torch.bfloat16
FFMPEG = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")


def wan_module(name):
    pkg = types.ModuleType("studio_wan")
    pkg.__path__ = [str(ROOT.parent.parent / "D_human_train/wan/modules")]
    sys.modules.setdefault(pkg.__name__, pkg)
    return importlib.import_module("studio_wan." + name)


@torch.inference_mode()
def encode(run):
    cases = read(run / "cases.json")
    if not any(c["kind"] == "产品描述" for c in cases):
        return
    encoder = wan_module("t5").T5EncoderModel(
        512,
        dtype=DTYPE,
        device="cuda",
        checkpoint_path=ENC / "models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path=str(ENC / "google/umt5-xxl"),
    )
    for i, c in enumerate(cases):
        if c["kind"] == "产品描述":
            torch.save(encoder([c["prompt"]], "cuda")[0].cpu(), run / f"text-{i}.pt")


def pad(t):
    return torch.nn.functional.pad(t, (0, 0, 0, 512 - len(t))).to("cuda", DTYPE)[None]


@torch.inference_mode()
def evaluate(run, rank, world):
    cfg = read(run / "config.json")
    cases = read(run / "cases.json")
    vae = wan_module("vae2_1").Wan2_1_VAE(
        vae_pth=str(ENC / "Wan2.1_VAE.pth"), dtype=DTYPE
    )
    neg = pad(
        torch.load(
            ROOT.parent / "outputs/l2-sport-fullbody/ab/negative.pt", weights_only=True
        )
    )
    versions = cfg.get("eval_versions", ["base", "sft"])
    if not versions or any(v not in ("base", "sft") for v in versions):
        raise ValueError("eval_versions must contain base and/or sft")
    for version in versions:
        model = (
            load_model(BASE).to(DTYPE)
            if version == "base"
            else WanTransformer3DModel.from_pretrained(
                run / "transformer", torch_dtype=DTYPE
            ).cuda()
        )
        model.eval().requires_grad_(False)
        for i in range(rank, len(cases), world):
            c = cases[i]
            generator = torch.Generator().manual_seed(c["seed"])
            initial = torch.randn([1, *c["shape"]], generator=generator).cuda()
            metrics = {}
            if "cache" in c:
                data = torch.load(
                    c["cache"], weights_only=True, mmap=True, map_location="cpu"
                )
                text = pad(data["text_embedding"])
                latent = data["video_latent"].cuda().float()[None]
                t = torch.tensor([500.0], device="cuda")
                pred = model(
                    ((latent + initial) * 0.5).to(DTYPE), t, text, return_dict=False
                )[0].float()
                metrics["flow_mse"] = torch.nn.functional.mse_loss(
                    pred, initial - latent
                ).item()
            else:
                text = pad(torch.load(run / f"text-{i}.pt", weights_only=True))
            scheduler = UniPCMultistepScheduler(
                prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
            )
            scheduler.set_timesteps(cfg["sampling_steps"], device="cuda")
            x = initial.clone()
            for t in scheduler.timesteps:
                pos = model(x.to(DTYPE), t.reshape(1), text, return_dict=False)[
                    0
                ].float()
                uncond = model(x.to(DTYPE), t.reshape(1), neg, return_dict=False)[
                    0
                ].float()
                x = scheduler.step(
                    uncond + 5 * (pos - uncond), t, x, return_dict=False
                )[0]
            frames = (
                vae.decode([x[0]])[0]
                .permute(1, 2, 3, 0)
                .add(1)
                .mul(127.5)
                .round()
                .byte()
                .cpu()
                .numpy()
            )
            h, w = frames.shape[1:3]
            dest = run / f"{i:02}-{version}.mp4"
            subprocess.run(
                [
                    FFMPEG,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    "24",
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-crf",
                    "17",
                    "-pix_fmt",
                    "yuv420p",
                    str(dest),
                ],
                input=frames.tobytes(),
                check=True,
            )
            metrics["frames"] = len(frames)
            write(run / f"{i:02}-{version}.json", metrics)
            if version == "sft":
                filters = "[0:v]pad=iw:ih+40:0:40,drawtext=text='Base':fontcolor=white:fontsize=24:x=(w-tw)/2:y=4[a];[1:v]pad=iw:ih+40:0:40,drawtext=text='SFT':fontcolor=white:fontsize=24:x=(w-tw)/2:y=4[b];[a][b]hstack[out]"
                subprocess.run(
                    [
                        FFMPEG,
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(run / f"{i:02}-base.mp4"),
                        "-i",
                        str(dest),
                        "-filter_complex",
                        filters,
                        "-map",
                        "[out]",
                        "-threads",
                        "2",
                        "-c:v",
                        "libx264",
                        "-movflags",
                        "+faststart",
                        str(run / f"pair-{i:02}.mp4"),
                    ],
                    check=True,
                )
            print(f"{version} case={i + 1}/{len(cases)}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run = Path(sys.argv[1])
    if sys.argv[2] == "encode":
        encode(run)
    else:
        evaluate(run, int(sys.argv[2]), int(sys.argv[3]))
