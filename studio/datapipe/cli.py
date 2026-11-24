"""Run heavy work with the fastvideo interpreter, outside the HTTP process."""

import argparse
import json
import os
import sys
from . import store


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    parser = argparse.ArgumentParser(description="视频数据质检与训练数据管线")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile")
    p.add_argument("jsonl")
    p.add_argument("--batch-id")
    p = sub.add_parser("ingest")
    p.add_argument("batch")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("sample")
    p.add_argument("batch")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra-l2", nargs="+")
    p.add_argument("--k", type=int, default=3)
    p = sub.add_parser("finish")
    p.add_argument("batch")
    p.add_argument("--override-reason")
    for command in ("recipe-parse", "plan", "build", "sample-extra"):
        p = sub.add_parser(command)
        p.add_argument("batch")
        p.add_argument("--recipe", help="JSON 文件；省略则从 stdin 读取 JSON")
    args = parser.parse_args()
    batch = None
    try:
        if args.command == "profile":
            from .profile import profile

            batch, _ = store.register(args.jsonl, args.batch_id)
            result = profile(batch)
            result = dict(batch_id=batch.name, rows=result["meta"]["rows"])
        else:
            batch = store.batch_path(args.batch)
            if args.command in ("sample", "ingest"):
                from .sample import sample

                if args.command == "ingest":
                    from .profile import profile

                    if not (batch / "profile-complete.json").exists():
                        profile(batch)
                    result = sample(batch, args.n, args.seed)
                else:
                    result = sample(batch, args.n, args.seed, args.extra_l2, args.k)
            elif args.command == "finish":
                from .qc import finish

                result = finish(batch, args.override_reason)
            else:
                body = store.read(args.recipe) if args.recipe else json.load(sys.stdin)
                if args.command == "sample-extra":
                    from .sample import sample

                    result = sample(
                        batch, extra_l2=body.get("extra_l2"), k_per_l2=body.get("k", 3)
                    )
                elif args.command == "recipe-parse":
                    from .recipe import parse

                    result = parse(body.get("prompt"), batch)
                    import secrets

                    parse_id = "parse-" + secrets.token_hex(8)
                    with store.lock(batch):
                        store.write(batch / "recipes" / (parse_id + ".json"), result)
                    result = dict(result, parse_id=parse_id)
                else:
                    from .build import plan, materialize

                    recipe = body.get("recipe", body)
                    if recipe.get("dataset") != batch.name:
                        raise ValueError("配方 dataset 与批次不一致")
                    if args.command == "plan":
                        result = dict(preview=[p.preview() for p in plan(recipe)])
                    else:
                        parse_id = body.get("parse_id")
                        if parse_id and (
                            not isinstance(parse_id, str)
                            or not store.ID.fullmatch(parse_id)
                            or parse_id in (".", "..")
                        ):
                            raise ValueError("无效的解析记录 ID")
                        runs = materialize(
                            recipe, parse_id, dispatch=body.get("dispatch") is True
                        )
                        from .build import RUNS

                        result = dict(
                            recipe_id=store.read(RUNS / runs[0] / "config.json")[
                                "recipe_id"
                            ],
                            runs=runs,
                        )
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        if batch and args.command == "ingest":
            with store.lock(batch):
                store.state(batch, "失败", error=str(exc))
        print(json.dumps(dict(error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
