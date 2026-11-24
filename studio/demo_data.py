"""Generate a synthetic video and manifest for exploring the QC workflow."""

import argparse
from pathlib import Path
import shutil
import subprocess

from datapipe.store import write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New directory for generated files")
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("Install FFmpeg with libx264")

    directory = args.directory.expanduser().resolve()
    directory.mkdir(parents=True)
    video = directory / "demo.mp4"
    subprocess.run(
        [
            ffmpeg, "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=320x180:rate=24:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(video),
        ],
        check=True,
    )
    manifest = directory / "demo.jsonl"
    ids = [f"demo{i:02d}_clip_001" for i in range(6)]
    write_jsonl(
        manifest,
        (
            dict(
                id=sample_id,
                video_path=str(video),
                caption="Synthetic moving test pattern.",
                width=320,
                height=180,
                fps=24,
                start=0,
                end=2,
                duration=2,
                screen_orientation="landscape",
                body_framing="half_body",
                l1_category_ids=["l1-01"],
                l1_category=["合成示例"],
                l2_category_ids=["l1-01-l2-01"],
                l2_category=["测试图案"],
            )
            for sample_id in ids
        ),
    )
    # Profiling uses index metadata to select the QC pool. Training still
    # requires real tensor files, which this demo does not generate.
    write_jsonl(
        directory / "index.jsonl",
        (
            dict(
                id=sample_id,
                cache=str(directory / f"{sample_id}.pt"),
                key=sample_id,
                config_hash="synthetic-demo",
                video_shape=[16, 12, 60, 104],
                num_frames=45,
                sampling=dict(l2_ids=["l1-01-l2-01"], l3=[]),
            )
            for sample_id in ids
        ),
    )
    print(manifest)


if __name__ == "__main__":
    main()
