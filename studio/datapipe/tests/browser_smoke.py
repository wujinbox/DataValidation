"""Optional browser acceptance; requires Playwright + Chromium, uses only synthetic data."""

# ruff: noqa: E402
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

STUDIO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDIO))
sys.path.insert(0, str(Path(__file__).parent))
from test_pipeline import Fixture
from datapipe import build, sample, store, recipe, queue
import server
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright


def main():
    fixture = Fixture()
    fixture.setUp()
    http = None
    try:
        ffmpeg = "/home/zkln/miniconda3/envs/fastvideo/bin/ffmpeg"
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=128x72:d=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(fixture.root / "video.mp4"),
            ],
            check=True,
        )
        sample.sample(fixture.batch, n=5)

        def heavy(command, batch, body, timeout=300):
            if command == "plan":
                return dict(preview=[p.preview() for p in build.plan(body["recipe"])])
            if command == "recipe-parse":
                return recipe.parse(body["prompt"], batch)
            if command == "sample-extra":
                return sample.sample(
                    batch, extra_l2=body["extra_l2"], k_per_l2=body["k"]
                )
            if command == "build":
                with patch.object(build, "validate_cache", return_value=None):
                    rids = build.materialize(
                        body["recipe"], dispatch=body.get("dispatch", False)
                    )
                return dict(
                    recipe_id=store.read(fixture.runs / rids[0] / "config.json")[
                        "recipe_id"
                    ],
                    runs=rids,
                )
            raise AssertionError(command)

        class Quiet(server.Handler):
            def log_message(self, *args):
                pass

        with (
            patch.object(server, "RUNS", fixture.runs),
            patch.object(server, "heavy", side_effect=heavy),
            patch.object(
                server, "probe", return_value=dict(cards=[dict(index=0, idle=True)])
            ),
            patch.object(
                server,
                "resources",
                return_value=[
                    dict(
                        host="0008",
                        cards=[dict(index=0, idle=True, used=0, total=80000)],
                    )
                ],
            ),
            patch.object(
                server, "launch", side_effect=AssertionError("测试不允许启动训练")
            ),
        ):
            http = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
            threading.Thread(target=http.serve_forever, daemon=True).start()
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
                page = browser.new_page(viewport=dict(width=1440, height=1100))
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(f"http://127.0.0.1:{http.server_port}/")
                assert page.locator(".tabs [data-tab]").count() == 3
                assert page.locator("[data-tab=recipe]").count() == 0
                assert page.locator("#training-page #dp-form").count() == 1
                assert page.locator("#training-page #jobs").count() == 1
                page.locator("[data-tab=datasets]").click()
                page.locator("#datasets-page .dp-batch").select_option(fixture.batch.name)
                page.wait_for_selector("#dp-report table")
                assert page.locator("#dp-report h2").count() == 10
                page.locator("[data-tab=qc]").click()
                page.locator("#dp-annotator").fill("自动化测试（合成数据）")
                page.wait_for_function(
                    'document.querySelector("#dp-video").readyState >= 1'
                )
                page.evaluate('document.querySelector("#dp-video").currentTime=1')
                page.wait_for_function(
                    'Math.abs(document.querySelector("#dp-video").currentTime-1)<.2'
                )
                page.locator("#dp-labels button").first.click()
                assert page.locator("#dp-labels .wrong").count() == 1
                page.locator("#dp-note").fill("合成样本审核")
                page.locator("#dp-pass").click()
                page.wait_for_function(
                    'document.querySelector("#dp-qc-progress").textContent.includes("已审 1 / 5")'
                )
                page.reload()
                page.locator("[data-tab=qc]").click()
                page.wait_for_function(
                    'document.querySelector("#dp-qc-progress").textContent.includes("已审 1 / 5")'
                )
                page.locator('[data-sample="1"]').click()
                for done in range(2, 6):
                    page.locator("#dp-pass").click()
                    page.wait_for_function(
                        f'document.querySelector("#dp-qc-progress").textContent.includes("已审 {done} / 5")'
                    )
                page.locator("#dp-finish").click()
                page.wait_for_function(
                    'document.querySelector("#dp-qc-progress").textContent.includes("可训练")'
                )
                page.locator("[data-tab=training]").click()
                page.locator("#dp-minimum").fill("1")
                page.locator("#dp-gpus").fill("0")
                page.locator("#dp-batch-size").fill("1")
                page.locator("#dp-products").fill("0")
                page.locator("#dp-category-search").fill("l1-01-l2-01")
                assert page.locator("#dp-categories label:visible").count() == 1
                page.locator("#dp-category-search").fill("")
                assert page.locator("#dp-categories label:visible").count() == 2
                page.locator("#dp-resource-refresh").click()
                page.wait_for_selector("#dp-resource-cards [data-gpu='0']")
                assert page.locator("#dp-resource-cards [data-gpu='0']").is_checked()
                page.locator("#dp-prompt").fill("训练手势")
                page.locator("#dp-parse").click()
                page.wait_for_function(
                    'document.querySelector("#dp-recipe-error").textContent.includes("请使用表单")'
                )
                page.locator("#dp-preview").click()
                page.wait_for_selector("#dp-preview-result table")
                assert not page.locator("#dp-submit").is_disabled()
                page.locator("#dp-lr").fill("0.000005")
                assert page.locator("#dp-submit").is_disabled()
                page.locator("#dp-preview").click()
                page.wait_for_function('!document.querySelector("#dp-submit").disabled')
                page.locator("#dp-submit").click()
                page.wait_for_function(
                    'document.querySelector("#dp-recipe-error").textContent.includes("已提交 1 个任务")'
                )
                assert len(queue.entries()) == 1
                page.wait_for_function('!document.querySelector("#training-editor").open')
                assert not page.locator("#training-editor").evaluate("el=>el.open")
                assert page.locator("#training-jobs").is_visible()
                assert "已提交 1 个训练任务" in page.locator("#note").inner_text()
                page.wait_for_selector('a[href$="/file/mix.json"]')
                page.locator("#training-editor > summary").click()
                assert page.locator("#dp-form").is_visible()
                assert errors == [], errors
                page.screenshot(path="/tmp/wan-pipeline-ui.png", full_page=True)
                # Width acceptance: all tabs should fit a phone without a page-level horizontal overflow.
                page.set_viewport_size(dict(width=390, height=844))
                for tab in ("datasets", "qc", "training"):
                    page.locator("[data-tab=" + tab + "]").click()
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= innerWidth+2"
                    ), tab
                browser.close()
                print(
                    "Browser acceptance passed: unified training/resources/category search/report/video seek/review persistence/gate/form/no-key/preview/submit/history/mobile"
                )
    finally:
        if http:
            http.shutdown()
            http.server_close()
        fixture.doCleanups()


if __name__ == "__main__":
    main()
