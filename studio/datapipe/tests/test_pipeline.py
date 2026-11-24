"""Fixtures live under /tmp. No GPU, real network, restart or publishing."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import gzip
import http.client
import json
import os
import threading
import unittest

import common
from datapipe import store, profile, sample, qc, recipe, build, cases, queue

A = "l1-01-l2-01"
B = "l1-01-l2-02"


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix="wan-pipeline-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ds = self.root / "datasets"
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for obj, key, value in [
            (store, "DATASETS", self.ds),
            (build, "RUNS", self.runs),
            (queue, "RUNS", self.runs),
            (queue, "QUEUE", self.root / "queue.json"),
        ]:
            self.stack.enter_context(patch.object(obj, key, value))
        self.source = self.root / "source.jsonl"
        self.index = self.root / "index.jsonl.gz"
        self.stack.enter_context(
            patch.dict(
                os.environ, {"WAN_INDEX": str(self.index), "WAN_QC_THRESHOLD": ".9"}
            )
        )
        self.raw = []
        self.index_rows = []
        for i in range(60):
            labels = [A, B] if i % 3 == 0 else [A] if i % 2 == 0 else [B]
            rid = f"video{i:03}_clip_001"
            self.raw.append(
                dict(
                    id=rid,
                    video_path=str(self.root / "video.mp4"),
                    duration=3.2,
                    start=0,
                    end=3.2,
                    width=1920,
                    height=1080,
                    fps=30,
                    screen_orientation="landscape",
                    body_framing="half_body",
                    review_required=i == 0,
                    body_framing_mixed=False,
                    global_ip_id=f"person{i}",
                    caption=f"A woman moves her arms and turns {i}.",
                    static_caption="A woman stands.",
                    dynamic_caption=f"She moves her arms {i}.",
                    l1_category_ids=["l1-01"],
                    l1_category=["身体动作"],
                    l2_category_ids=labels,
                    l2_category=["类别甲" if x == A else "类别乙" for x in labels],
                    dataset_source="fixture",
                    language_review=[],
                )
            )
            self.index_rows.append(
                dict(
                    id=rid,
                    key=f"key{i}",
                    config_hash="fixture",
                    cache=str(self.root / f"{i}.pt"),
                    num_frames=77,
                    video_shape=[16, 20, 60, 104],
                    sampling=dict(l2_ids=labels, pool="fixture", l3=[]),
                )
            )
        self.rewrite()
        (self.root / "video.mp4").write_bytes(bytes(range(256)) * 4)
        self.batch, _ = store.register(str(self.source))
        with patch.object(profile, "charts"):
            self.stats = profile.profile(self.batch)

    def rewrite(self):
        store.write_jsonl(self.source, self.raw)
        with gzip.open(self.index, "wt") as f:
            for row in self.index_rows:
                f.write(json.dumps(row) + "\n")

    def reviewed(self, n=10, failed=0):
        s = sample.sample(self.batch, n=n)
        for i, item in enumerate(s["items"]):
            qc.save(
                self.batch,
                dict(
                    idx=item["idx"],
                    id=item["id"],
                    annotator="测试审核人",
                    verdict="fail" if i < failed else "pass",
                    issues=[],
                ),
            )
        return qc.finish(self.batch)

    def r(self, **kwargs):
        r = dict(
            dataset=self.batch.name,
            categories=[A, B],
            mode="mixed",
            min_category_size=1,
            train=dict(host="0008", gpus=[0], batch=2),
            eval=dict(products=0),
        )
        r.update(kwargs)
        return r


class ProfileTests(Fixture):
    def test_stats_and_ten_sections(self):
        self.assertEqual(self.stats["meta"]["rows"], 60)
        self.assertEqual(self.stats["trainability"]["pool"], 60)
        self.assertEqual(len(self.stats["categories"]["l2"]), 2)
        report = (self.batch / "分布报告.md").read_text()
        for i in range(1, 11):
            self.assertIn(f"## {i}.", report)
        self.assertIn("配方默认排除", report)

    def test_tolerant_fields_duplicates_and_missing_cache(self):
        raw = self.raw[:3] + [
            self.raw[0],
            dict(id="missing-cache", odd_bool=True, odd_number=5, odd_list=["x"]),
            dict(
                id="bad-types", width=[], l2_category_ids="bad", review_required="yes"
            ),
        ]
        other = self.root / "other.jsonl"
        store.write_jsonl(other, raw)
        with other.open("a") as f:
            f.write("{invalid\n[]\n")
        batch, _ = store.register(str(other))
        with patch.object(profile, "charts"):
            stats = profile.profile(batch)
        self.assertEqual(stats["meta"]["rows"], 8)
        self.assertEqual(stats["meta"]["invalid_rows"], 2)
        self.assertEqual(stats["meta"]["dup_ids"], 1)
        self.assertEqual(stats["trainability"]["pool"], 3)
        self.assertEqual(stats["auto_fields"]["odd_number"]["type"], "numeric")
        self.assertEqual(stats["auto_fields"]["odd_bool"]["type"], "categorical")
        self.assertEqual(stats["auto_fields"]["odd_list"]["dist"], {"x": 1})
        self.assertGreater(stats["meta"]["missing_fields"]["caption"], 0)
        self.assertIn("width", stats["meta"]["invalid_types"])
        self.assertEqual(stats["trainability"]["cache_miss"], 2)

    def test_index_invalidation_and_shapes(self):
        original = common.load_index_by_id()
        self.index_rows[0]["num_frames"] = 76
        self.rewrite()
        changed = common.load_index_by_id()
        self.assertIsNot(original, changed)
        self.assertFalse(common.trainable(changed[self.raw[0]["id"]]))
        self.assertTrue(common.trainable(self.index_rows[1]))

    def test_registration_and_traversal(self):
        same, created = store.register(str(self.source))
        self.assertEqual(same, self.batch)
        self.assertFalse(created)
        for value in ("..", ".", "../abc", "a/b", "x" * 121):
            with self.assertRaises(ValueError):
                store.batch_path(value)
        with self.assertRaises(ValueError):
            store.register("relative.jsonl")

    def test_atomic_parallel_writes(self):
        path = self.root / "shared.json"
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda x: store.write(path, dict(number=x)), range(40)))
        self.assertIn(store.read(path)["number"], range(40))


class SamplingQCTests(Fixture):
    def test_determinism_and_diversity(self):
        first = sample.sample(self.batch)
        digest = common.sha256_file(self.batch / "qc/sample.json")
        second = sample.sample(self.batch)
        self.assertEqual(first, second)
        self.assertEqual(digest, common.sha256_file(self.batch / "qc/sample.json"))
        self.assertEqual(len({x["global_ip_id"] for x in first["items"]}), 20)
        self.assertEqual(len({x["source_group"] for x in first["items"]}), 20)
        self.assertTrue(any(x["review_required"] for x in first["items"]))
        self.assertEqual(set(first["coverage"]["covered_l2"]), {A, B})
        for x in first["items"]:
            self.assertEqual(
                [c["name"] for c in x["l2"]],
                ["类别甲" if c["id"] == A else "类别乙" for c in x["l2"]],
            )

    def test_gate_override_and_latest_review(self):
        result = self.reviewed(failed=2)
        self.assertEqual(result["gate"], "待处理")
        self.assertEqual(result["pass_rate"], 0.8)
        result = qc.finish(self.batch, "测试放行理由")
        self.assertEqual(result["gate"], "可训练")
        item = store.read(self.batch / "qc/sample.json")["items"][0]
        qc.save(
            self.batch, dict(idx=0, id=item["id"], annotator="第二人", verdict="pass")
        )
        self.assertIsNone(store.read(self.batch / "qc/summary.json"))
        self.assertEqual(qc.latest(self.batch)[0]["annotator"], "第二人")
        self.assertEqual(qc.finish(self.batch)["gate"], "可训练")

    def test_incomplete_invalid_and_extra(self):
        sample.sample(self.batch, n=5)
        with self.assertRaisesRegex(ValueError, "全部审完"):
            qc.finish(self.batch)
        item = store.read(self.batch / "qc/sample.json")["items"][0]
        for update in [
            dict(verdict="bad"),
            dict(id="bad"),
            dict(issues=["invalid"]),
            dict(annotator=""),
        ]:
            v = dict(idx=0, id=item["id"], annotator="tester", verdict="pass")
            v.update(update)
            with self.assertRaises(ValueError):
                qc.save(self.batch, v)
        self.reviewed(n=5)
        old = store.read(self.batch / "qc/sample.json")["items"]
        result = sample.sample(self.batch, extra_l2=[A], k_per_l2=3)
        self.assertEqual(result["items"][:5], old)
        self.assertEqual(len(result["items"]), 8)
        self.assertEqual(len(qc.latest(self.batch)), 5)
        self.assertIsNone(store.read(self.batch / "qc/summary.json"))
        with self.assertRaises(ValueError):
            build.plan(self.r())

    def test_no_resampling_reviewed_data(self):
        self.reviewed()
        with self.assertRaisesRegex(ValueError, "已有审核记录"):
            sample.sample(self.batch, n=11)

    def test_invalid_threshold_and_tampered_summary(self):
        self.reviewed()
        with patch.dict(os.environ, {"WAN_QC_THRESHOLD": "nan"}):
            with self.assertRaises(ValueError):
                qc.finish(self.batch)
        with (self.batch / "qc/reviews.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "已变化"):
            build.plan(self.r())


class RecipeBuildTests(Fixture):
    def test_validation_boundaries(self):
        for bad in [
            self.r(train=dict(lr=1)),
            self.r(categories=["missing"]),
            self.r(mode="per_category", min_category_size=200),
            self.r(mix={"default": float("nan")}),
            self.r(train=dict(gpus=[True])),
            self.r(train=dict(gpus=[0, 0])),
            self.r(filters=dict(exclude_review_required=1)),
            self.r(eval=dict(max_cases=0)),
        ]:
            with self.assertRaises(ValueError):
                recipe.validate(bad, self.stats)
        valid, _ = recipe.validate(self.r(), self.stats)
        self.assertEqual(valid["train"]["lr"], 3e-6)

    def test_plan_coverage_exclusion_and_steps(self):
        summary = self.reviewed(failed=1)
        p = build.plan(self.r())[0]
        self.assertFalse(set(summary["excluded_ids"]) & {r["id"] for r in p.rows})
        self.assertFalse(p.heldout_groups & {r["source_group"] for r in p.rows})
        self.assertEqual({x for c in p.cases for x in c["covered_l2"]}, {A, B})
        self.assertEqual(p.preview()["epoch_steps"], len(p.rows) // 2)
        self.assertFalse(any(self.runs.iterdir()))

    def test_weighting_zero_overlap_target_and_determinism(self):
        rows = [
            dict(id="ab", l2_ids=[A, B]),
            dict(id="a", l2_ids=[A]),
            dict(id="b", l2_ids=[B]),
        ]
        first = build._weighted(rows, [B], {"default": 1, A: 0, B: 2}, 0, 42)
        self.assertEqual(Counter(x["id"] for x in first), {"ab": 2, "b": 2})
        self.assertEqual(
            first, build._weighted(rows, [B], {"default": 1, A: 0, B: 2}, 0, 42)
        )
        self.assertEqual(len(build._weighted(rows, [B], {"default": 1}, 17, 42)), 17)
        self.assertEqual(len(build._weighted(rows, [B], {"default": 1}, 1, 42)), 1)

    def test_mixed_fails_uncovered_and_last_training_group(self):
        self.reviewed()
        rows = list(store.jsonl(self.batch / "pool.jsonl"))
        for r in rows:
            r["l2_ids"] = [A] if int(r["id"][5:8]) % 2 == 0 else [B]
        store.write_jsonl(self.batch / "pool.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "测试未覆盖"):
            build.plan(self.r(eval=dict(max_cases=1, products=0)))
        for r in rows:
            if B in r["l2_ids"]:
                r["source_group"] = "single-b-source"
        store.write_jsonl(self.batch / "pool.jsonl", rows)
        with self.assertRaisesRegex(ValueError, B):
            build.plan(self.r())

    def test_filter_and_fragmented_batch_rejected(self):
        self.reviewed()
        with self.assertRaisesRegex(ValueError, "过滤后"):
            build.plan(self.r(filters=dict(min_short_side=5000)))
        with self.assertRaisesRegex(ValueError, "batch"):
            build.plan(self.r(train=dict(gpus=list(range(8)), batch=6), categories=[A]))

    def test_per_category_expansion_and_materialization(self):
        self.reviewed()
        r = self.r(mode="per_category", eval=dict(heldout_per_category=3, products=0))
        self.assertEqual(len(build.plan(r)), 2)
        with patch.object(build, "validate_cache", return_value=None):
            rids = build.materialize(r)
        self.assertEqual(len(rids), 2)
        for rid in rids:
            run = self.runs / rid
            cfg = store.read(run / "config.json")
            mix = store.read(run / "mix.json")
            self.assertTrue(cfg["prepared_data"])
            self.assertFalse(cfg["dispatch_authorized"])
            self.assertEqual(
                mix["train_sha256"], common.sha256_file(run / "train.jsonl")
            )
            self.assertTrue((run / "test_prompts.jsonl").exists())
            rows = list(store.jsonl(run / "train.jsonl"))
            counts = Counter(
                x
                for row in rows
                for x in next(r for r in self.raw if r["id"] == row["id"])[
                    "l2_category_ids"
                ]
            )
            for c in mix["per_l2"]:
                self.assertEqual(c["ratio_in_train"], counts[c["id"]] / len(rows))
            from publish import data_tables

            report = data_tables(run, store.read(run / "cases.json"))
            self.assertIn("数据配比", report)
            self.assertIn("L2 → 测试样例", report)
        self.assertEqual(len((self.batch / "runs.csv").read_text().splitlines()), 3)

    def test_bad_cache_aborts_before_any_run(self):
        self.reviewed()
        with patch.object(build, "validate_cache", return_value="损坏"):
            with self.assertRaisesRegex(ValueError, "缓存抽检失败"):
                build.materialize(self.r())
        self.assertFalse(any(self.runs.iterdir()))

    def test_llm_privacy_retry_and_no_key(self):
        with patch.dict(os.environ, {"WAN_RECIPE_LLM": "openai", "OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "请使用表单"):
                recipe.parse("训练手势", self.batch)
        replies = [json.dumps(self.r(train=dict(lr=1))), json.dumps(self.r())]
        with (
            patch.dict(
                os.environ,
                {"WAN_RECIPE_LLM": "openai", "OPENAI_API_KEY": "fixture-key"},
            ),
            patch.object(recipe, "_request", side_effect=replies) as request,
            patch.object(build, "plan", return_value=[]),
        ):
            out = recipe.parse("训练手势", self.batch)
            self.assertEqual(len(out["llm_raw"]), 2)
            system = request.call_args_list[0].args[2]
            self.assertNotIn(str(self.source), system)
            self.assertNotIn(self.raw[0]["caption"], system)
            self.assertNotIn("person0", system)

    def test_cases_reserve_training_source(self):
        self.reviewed()
        p = build.plan(self.r())[0]
        self.assertTrue(p.cases)
        self.assertEqual(cases.products([], [], {}, 0, 42), [])

    def test_fragmented_shapes_with_enough_total_rows(self):
        self.reviewed()
        index = {r["id"]: dict(r) for r in self.index_rows}
        for i, row in enumerate(index.values()):
            t = 12 + i % 14
            row["video_shape"] = [16, t, 60, 104]
            row["num_frames"] = 4 * t - 3
        with patch.object(build, "load_index_by_id", return_value=index):
            with self.assertRaisesRegex(ValueError, "完整 batch"):
                build.plan(self.r(train=dict(gpus=[0], batch=6)))

    def test_pool_drift_and_source_version_rejected(self):
        self.reviewed()
        with self.source.open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "源数据已变化"):
            build.plan(self.r())

    def test_llm_provider_wire_format(self):
        from io import BytesIO

        class Response(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        for provider, data in [
            ("openai", {"choices": [{"message": {"content": "{}"}}]}),
            ("anthropic", {"content": [{"type": "text", "text": "{}"}]}),
        ]:
            with (
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test",
                        "ANTHROPIC_API_KEY": "test",
                        "WAN_RECIPE_BASE_URL": "https://example.invalid/v1",
                    },
                ),
                patch.object(
                    recipe, "urlopen", return_value=Response(json.dumps(data).encode())
                ) as request,
            ):
                self.assertEqual(
                    recipe._request(
                        provider,
                        "configured-model",
                        "目录",
                        [{"role": "user", "content": "配方"}],
                        2,
                    ),
                    "{}",
                )
                req = request.call_args.args[0]
                body = json.loads(req.data)
                self.assertEqual(body["model"], "configured-model")
                self.assertIn("/v1/", req.full_url)
                self.assertEqual(
                    body.get("system")
                    if provider == "anthropic"
                    else body["messages"][0]["content"],
                    "目录",
                )


class QueueTests(Fixture):
    def queued(self):
        ids = ["20261116-120000-a00001", "20261116-120000-a00002"]
        for rid in ids:
            run = self.runs / rid
            run.mkdir()
            store.write(run / "config.json", dict(host="0008", gpus=[0]))
            store.write(run / "state.json", dict(stage="数据已核验"))
        queue.enqueue("recipe-test", ids)
        return ids

    def test_serial_reservations_and_cancellation(self):
        ids = self.queued()
        calls = []
        with patch.object(queue, "_alive", return_value=True):
            queue.tick(lambda r: calls.append(r.name))
            queue.tick(lambda r: calls.append(r.name))
            self.assertEqual(calls, [ids[0]])
            self.assertTrue(queue.conflicts("0008", [0]))
            store.write(self.runs / ids[0] / "state.json", dict(stage="完成"))
            queue.tick(lambda r: calls.append(r.name))
            self.assertEqual(calls, ids)
            store.write(self.runs / ids[1] / "state.json", dict(stage="完成"))
            queue.tick(lambda r: None)
            self.assertFalse(queue.conflicts("0008", [0]))

    def test_recovery_does_not_duplicate_and_waiting_cancel(self):
        ids = self.queued()
        self.assertTrue(queue.cancel_waiting(self.runs / ids[1]))
        calls = []
        queue.tick(lambda r: calls.append(r.name))
        with patch.object(queue, "_alive", return_value=False):
            queue.tick(lambda r: calls.append(r.name))
        self.assertEqual(calls, [ids[0]])
        self.assertEqual(
            store.read(self.runs / ids[0] / "state.json")["stage"], "恢复待确认"
        )
        self.assertTrue(queue.conflicts("0008", [0]))

    def test_interrupted_enqueue_recovers_prepared_state(self):
        ids = self.queued()
        # Queue intent was committed, but per-run status writes were interrupted.
        for rid in ids:
            store.write(self.runs / rid / "state.json", dict(stage="数据已核验"))
        calls = []
        queue.tick(lambda r: calls.append(r.name))
        self.assertEqual(calls, [ids[0]])

    def test_manual_materialization_is_not_automatically_launched(self):
        self.reviewed()
        with patch.object(build, "validate_cache", return_value=None):
            build.materialize(self.r())
        queue.recover()
        self.assertEqual(queue.entries(), [])

    def test_changed_qc_blocks_queued_dispatch(self):
        self.reviewed()
        with patch.object(build, "validate_cache", return_value=None):
            rids = build.materialize(self.r(), dispatch=True)
        recipe_id = store.read(self.runs / rids[0] / "config.json")["recipe_id"]
        queue.enqueue(recipe_id, rids)
        store.invalidate_qc(self.batch)
        calls = []
        queue.tick(lambda r: calls.append(r.name))
        self.assertEqual(calls, [])
        self.assertEqual(
            store.read(self.runs / rids[0] / "state.json")["stage"], "失败"
        )

    def test_recipe_recovery_preserves_declared_order(self):
        self.reviewed()
        with patch.object(build, "validate_cache", return_value=None):
            rids = build.materialize(self.r(mode="per_category"), dispatch=True)
        cfg = store.read(self.runs / rids[0] / "config.json")
        record_path = self.batch / "recipes" / (cfg["recipe_id"] + ".json")
        record = store.read(record_path)
        record["runs"] = list(reversed(rids))
        store.write(record_path, record)
        queue.recover()
        self.assertEqual(queue.entries()[0]["runs"], list(reversed(rids)))


class APITests(Fixture):
    def setUp(self):
        super().setUp()
        import server

        self.server_module = server
        self.stack.enter_context(patch.object(server, "RUNS", self.runs))
        self.stack.enter_context(patch.object(server, "ROOT", self.root))
        self.stack.enter_context(
            patch.object(
                server, "probe", side_effect=AssertionError("不允许真实集群调用")
            )
        )
        self.stack.enter_context(
            patch.object(server, "launch", side_effect=AssertionError("不允许真实训练"))
        )
        self.stack.enter_context(patch.object(server, "resources", return_value=[]))
        self.sample = sample.sample(self.batch, n=5)
        from http.server import ThreadingHTTPServer

        class Quiet(server.Handler):
            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)

    def request(self, path, data=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.http.server_port, timeout=5
        )
        raw = json.dumps(data).encode() if data is not None else None
        base = {
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
            "X-Wan-Studio": "1",
        }
        base.update(headers or {})
        connection.request(
            "POST" if data is not None else "GET", path, body=raw, headers=base
        )
        response = connection.getresponse()
        code = response.status
        head = dict(response.getheaders())
        body = response.read()
        connection.close()
        return code, head, body

    def test_video_range_suffix_invalid_and_whitelist(self):
        url = "/api/datasets/" + self.batch.name + "/video/0"
        code, head, body = self.request(url, headers={"Range": "bytes=10-19"})
        self.assertEqual(code, 206)
        self.assertEqual(body, bytes(range(10, 20)))
        self.assertEqual(head["Content-Range"], "bytes 10-19/1024")
        code, _, body = self.request(url, headers={"Range": "bytes=-10"})
        self.assertEqual(code, 206)
        self.assertEqual(body, (self.root / "video.mp4").read_bytes()[-10:])
        for value in ("bytes=2000-", "bytes=5-2", "bytes=-0", "bytes=0-1,3-4", "junk"):
            self.assertEqual(self.request(url, headers={"Range": value})[0], 416)
        self.assertEqual(self.request(url.replace("/0", "/999"))[0], 404)
        self.assertEqual(self.request("/api/datasets/../server.py")[0], 404)
        self.assertEqual(
            self.request(
                "/api/datasets/" + self.batch.name + "/charts/../../server.py"
            )[0],
            404,
        )

    def test_security_body_limits_and_review_persistence(self):
        url = "/api/datasets/" + self.batch.name + "/reviews"
        item = self.sample["items"][0]
        body = dict(
            idx=0, id=item["id"], annotator="接口测试", verdict="pass", note="x" * 5000
        )
        self.assertEqual(self.request(url, body, {"X-Wan-Studio": ""})[0], 403)
        self.assertEqual(
            self.request(url, body, {"Origin": "https://evil.invalid"})[0], 403
        )
        self.assertEqual(self.request(url, body)[0], 200)
        self.assertEqual(self.request("/api/jobs", dict(padding="x" * 5000))[0], 400)
        self.assertEqual(self.request(url, dict(body, note="x" * 70000))[0], 400)
        code, _, result = self.request("/api/datasets/" + self.batch.name)
        self.assertEqual(code, 200)
        self.assertEqual(len(json.loads(result)["reviews"]), 1)

    def test_artifact_whitelist(self):
        rid = "20261116-120000-a00001"
        run = self.runs / rid
        run.mkdir()
        store.write(run / "mix.json", {"ok": True})
        self.assertEqual(self.request("/api/jobs/" + rid + "/file/mix.json")[0], 200)
        self.assertEqual(self.request("/api/jobs/" + rid + "/file/config.json")[0], 404)

    def test_prepared_submit_checks_reservations(self):
        self.reviewed(n=5)
        server = self.server_module

        def materialize(command, batch, body, timeout=300):
            self.assertEqual(command, "build")
            with patch.object(build, "validate_cache", return_value=None):
                rids = build.materialize(body["recipe"], dispatch=True)
            cfg = store.read(self.runs / rids[0] / "config.json")
            return dict(recipe_id=cfg["recipe_id"], runs=rids)

        with (
            patch.object(
                server, "probe", return_value={"cards": [{"index": 0, "idle": True}]}
            ),
            patch.object(server, "heavy", side_effect=materialize),
        ):
            code, _, raw = self.request(
                "/api/datasets/" + self.batch.name + "/recipe/submit",
                {"recipe": self.r()},
            )
            self.assertEqual(code, 201, raw)
            self.assertEqual(len(json.loads(raw)["runs"]), 1)
            self.assertEqual(
                self.request(
                    "/api/datasets/" + self.batch.name + "/recipe/submit",
                    {"recipe": self.r()},
                )[0],
                400,
            )
        self.assertEqual(len(queue.entries()), 1)


if __name__ == "__main__":
    unittest.main()
