"""HTTP-level smoke test for the FastAPI app (no server process needed)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402
from app.samples import SAMPLES  # noqa: E402

client = TestClient(app)


def test_index_and_health():
    assert client.get("/healthz").json() == {"status": "ok"}
    html = client.get("/")
    assert html.status_code == 200
    assert "共享词嵌入" in html.text
    assert client.get("/static/index.html").status_code == 200


def test_samples_endpoint_and_each_sample_passes():
    body = client.get("/api/samples").json()
    assert len(body["samples"]) == len(SAMPLES) == 4
    for sample in body["samples"]:
        # /api/run 严格拒绝未知字段：带 merge 配置的样例只走合并试算端点。
        run_payload = {k: v for k, v in sample["payload"].items() if k != "merge"}
        resp = client.post("/api/run", json=run_payload)
        assert resp.status_code == 200, (sample["id"], resp.text)
        report = resp.json()
        assert report["ok"], (sample["id"], report["errors"], report["split"])
        assert report["errors"]["loss_abs_diff"] <= 1e-8
        assert report["errors"]["grad_max_abs_diff"] <= 1e-8
        assert report["errors"]["weight_max_abs_diff"] <= 1e-8


def test_merge_trial_endpoint_happy_path():
    payload = {
        "seed": 20240517,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [
            {"name": "A", "tokens": [[4, 4], [4, 4], [7, 4], [4, 7]]},
            {"name": "B", "tokens": [[4, 4], [9, 9], [9, -100], [4, 9]]},
        ],
        "merge": {"threshold": 0.05, "groups": [[4, 7, 9], [10, 11]]},
    }
    resp = client.post("/api/merge-trial", json=payload)
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert report["ok"] is True
    r = report["result"]
    assert r["mergeable_count"] + r["revoked_count"] == 2
    # 撤销组必须带超限词对与距离；可合并组不得有超限词对。
    for g in r["revoked_groups"]:
        assert any(p["exceeds"] for p in g["pairs"])
    for g in r["mergeable_groups"]:
        assert all(not p["exceeds"] for p in g["pairs"])
    assert report["memory"]["max_vocab_chunk"] == 128
    assert report["memory"]["probability_normalization"] == "full_vocab_softmax"


def test_merge_trial_sample_runs_end_to_end():
    sample = next(s for s in SAMPLES if s["id"] == "merge-trial-demo")
    resp = client.post("/api/merge-trial", json=sample["payload"])
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert report["config"]["candidate_groups"] == 2
    assert report["config"]["candidate_pairs"] == 4
    assert report["result"]["mergeable_count"] + report["result"]["revoked_count"] == 2


def test_merge_trial_validation_returns_422():
    cases = [
        # 组间重叠
        {"seed": 1, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}],
         "merge": {"threshold": 0.1, "groups": [[1, 2], [2, 3]]}},
        # 阈值越界
        {"seed": 1, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}],
         "merge": {"threshold": 1.5, "groups": [[1, 2]]}},
        # 组成员数量越界
        {"seed": 1, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}],
         "merge": {"threshold": 0.1, "groups": [[1]]}},
        # 缺少 merge
        {"seed": 1, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
        # 引擎层错误（全忽略目标）
        {"seed": 1, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, -100]]}],
         "merge": {"threshold": 0.1, "groups": [[1, 2]]}},
    ]
    for payload in cases:
        resp = client.post("/api/merge-trial", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["detail"]


def test_invalid_requests_return_422():
    cases = [
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, -100]]}]},
        {"seed": 0, "scale": 200, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 128,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}] * 5},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]] * 129}]},
    ]
    for payload in cases:
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["detail"]


def test_flat_tokens_shorthand():
    resp = client.post("/api/run", json={
        "seed": 1, "scale": 1, "vocab_size": 256, "tokens": [[1, 2], [3, -100]],
    })
    assert resp.status_code == 200
    assert resp.json()["ok"]
