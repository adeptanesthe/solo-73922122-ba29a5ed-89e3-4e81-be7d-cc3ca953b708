"""合并试算（merge trial）的数值、校验、撤销流程与内存约束测试。

运行： python -m pytest tests/test_merge.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine import (  # noqa: E402
    MAX_VOCAB_CHUNK,
    ValidationError,
    make_weight,
    run_chunked,
)
import app.merge as merge_mod  # noqa: E402
from app.merge import (  # noqa: E402
    MAX_GROUPS,
    MAX_GROUP_MEMBERS,
    parse_merge_payload,
    run_merge_trial,
)


BASE = {
    "seed": 20240517,
    "scale": 1.0,
    "vocab_size": 256,
    "microbatches": [
        {"name": "A", "tokens": [[4, 4], [4, 4], [7, 4], [4, 7]]},
        {"name": "B", "tokens": [[4, 4], [9, 9], [9, -100], [4, 9]]},
    ],
}


def _payload(groups, threshold=0.05, base=None):
    p = dict(base or BASE)
    p["merge"] = {"threshold": threshold, "groups": groups}
    return p


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "merge",
    [
        {"threshold": 0.05, "groups": []},                     # 0 组
        {"threshold": 0.05, "groups": [[1, 2]] * (MAX_GROUPS + 1)},
        {"threshold": 0.05, "groups": [[1]]},                  # 成员太少
        {"threshold": 0.05, "groups": [[1] * (MAX_GROUP_MEMBERS + 1)]},
        {"threshold": -0.01, "groups": [[1, 2]]},              # 阈值越界
        {"threshold": 1.01, "groups": [[1, 2]]},
        {"threshold": 0.05, "groups": [[1, 2], [2, 3]]},       # 组间重叠
        {"threshold": 0.05, "groups": [[1, 1]]},               # 组内重复
        {"threshold": 0.05, "groups": [[1, 256]]},             # 词号越界
        {"threshold": "x", "groups": [[1, 2]]},
        {"threshold": 0.05, "groups": [[1, "x"]]},
        {"threshold": 0.05, "groups": "not-a-list"},
    ],
)
def test_invalid_merge_payloads_rejected(merge):
    with pytest.raises(ValidationError):
        parse_merge_payload(_payload(merge["groups"], merge["threshold"]))


def test_unknown_fields_rejected():
    p = _payload([[1, 2]])
    p["merge"]["bogus"] = 1
    with pytest.raises(ValidationError):
        parse_merge_payload(p)
    p2 = _payload([[1, 2]])
    p2["bogus"] = 1
    with pytest.raises(ValidationError):
        parse_merge_payload(p2)


def test_boundary_shapes_accepted():
    # 阈值 0 / 1；1 组 2 成员；4 组各 8 成员且互不重叠。
    parse_merge_payload(_payload([[1, 2]], threshold=0.0))
    parse_merge_payload(_payload([[1, 2]], threshold=1.0))
    groups = [list(range(2 + i * 8, 10 + i * 8)) for i in range(4)]
    spec, req = parse_merge_payload(_payload(groups, threshold=0.5))
    assert len(req.groups) == 4
    assert all(len(g) == 8 for g in req.groups)
    assert req.threshold == 0.5
    assert spec.vocab_size == 256


def test_engine_level_validation_still_applies():
    # 全部目标被 -100：引擎拒绝，合并试算同样不能运行。
    p = {
        "seed": 1,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [{"tokens": [[1, -100], [2, -100]]}],
        "merge": {"threshold": 0.1, "groups": [[1, 2]]},
    }
    with pytest.raises(ValidationError):
        parse_merge_payload(p)


# ---------------------------------------------------------------------------
# 数值正确性：对照无分块的稠密暴力实现
# ---------------------------------------------------------------------------
def _dense_tv_reference(weight, members):
    """完整 [m, V] softmax 后构造镜像分布，逐输出项算 0.5*L1。"""
    vocab = weight.shape[0]
    probs = torch.stack(
        [torch.softmax(weight[m] @ weight.t(), dim=0) for m in members]
    )
    member_idx = torch.tensor(list(members))
    others = torch.tensor([k for k in range(vocab) if k not in set(members)])
    group_item = probs[:, member_idx].sum(dim=1)
    other_items = probs[:, others]
    out = {}
    m = len(members)
    for i in range(m):
        for j in range(i + 1, m):
            tv = 0.5 * (
                abs(group_item[i] - group_item[j])
                + (other_items[i] - other_items[j]).abs().sum()
            )
            out[(members[i], members[j])] = float(tv.item())
    return out


def test_chunked_distances_match_dense_brute_force():
    payload = _payload([[4, 7, 9], [10, 11]], threshold=0.05)
    report = run_merge_trial(payload)
    spec, req = parse_merge_payload(payload)
    weight = run_chunked(spec).updated_weight

    got = {}
    for g in report["result"]["mergeable_groups"] + report["result"]["revoked_groups"]:
        for p in g["pairs"]:
            got[(p["a"], p["b"])] = p["distance"]

    worst = 0.0
    for members in req.groups:
        ref = _dense_tv_reference(weight, list(members))
        for key, ref_tv in ref.items():
            worst = max(worst, abs(got[key] - ref_tv))
    assert worst < 1e-12, worst


def test_distances_use_updated_weight_not_base():
    # 重复目标让单步 SGD 对相关行的改动足够大，base/updated 的 TV 可区分。
    p = {
        "seed": 3,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [
            {"tokens": [[1, 2], [1, 2], [1, 2], [3, 4]]}
        ],
        "merge": {"threshold": 1.0, "groups": [[1, 3]]},
    }
    spec, _ = parse_merge_payload(p)
    base = make_weight(spec.vocab_size, spec.seed, spec.scale)
    updated = run_chunked(spec).updated_weight
    assert not torch.equal(base, updated)
    report = run_merge_trial(p)
    pair = report["result"]["mergeable_groups"][0]["pairs"][0]
    ref_updated = _dense_tv_reference(updated, [1, 3])
    ref_base = _dense_tv_reference(base, [1, 3])
    # 报告值必须等于“更新后 W'”的距离，而不是初始 W 的距离。
    assert abs(pair["distance"] - ref_updated[(1, 3)]) < 1e-12
    assert abs(pair["distance"] - ref_base[(1, 3)]) > 1e-6


def test_probabilities_normalized_over_full_vocab():
    payload = _payload([[1, 2]])
    spec, _ = parse_merge_payload(payload)
    weight = run_chunked(spec).updated_weight
    # 全词表 softmax 概率和为 1（分块 logsumexp 与稠密 softmax 一致）。
    dense = torch.softmax(weight[1] @ weight.t(), dim=0)
    assert abs(float(dense.sum()) - 1.0) < 1e-14
    report = run_merge_trial(payload)
    pair = report["result"]["mergeable_groups"][0]["pairs"][0]
    assert 0.0 <= pair["distance"] <= 1.0


# ---------------------------------------------------------------------------
# 撤销流程
# ---------------------------------------------------------------------------
def test_threshold_zero_revokes_all_positive_distance_groups():
    report = run_merge_trial(_payload([[1, 2], [3, 4, 5]], threshold=0.0))
    assert report["result"]["mergeable_count"] == 0
    assert report["result"]["revoked_count"] == 2
    for g in report["result"]["revoked_groups"]:
        assert g["exceeds_threshold"]
        assert all(p["exceeds"] for p in g["pairs"] if p["distance"] > 0)


def test_threshold_one_keeps_every_group():
    report = run_merge_trial(_payload([[1, 2], [3, 4, 5]], threshold=1.0))
    assert report["result"]["revoked_count"] == 0
    assert report["result"]["mergeable_count"] == 2
    assert all(not p["exceeds"] for g in report["result"]["mergeable_groups"]
               for p in g["pairs"])


def test_iterative_revocation_rechecks_remaining_groups():
    base = {
        "seed": 11,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [{"tokens": [[1, 2], [3, 4], [5, 6], [7, 8]]}],
    }
    # 0.005 在该数据上出现“部分撤销 + 第二轮重检”。
    report = run_merge_trial(
        _payload([[1, 2], [3, 4], [5, 6], [7, 8]], threshold=0.005, base=base)
    )
    assert report["result"]["iterations"] >= 2
    assert report["result"]["mergeable_count"] >= 1
    assert report["result"]["revoked_count"] >= 1
    # 最终保留组里不能再有任何超限词对。
    for g in report["result"]["mergeable_groups"]:
        assert all(not p["exceeds"] for p in g["pairs"])
    # 每个撤销组都给出了超限词对和距离。
    for g in report["result"]["revoked_groups"]:
        bad = [p for p in g["pairs"] if p["exceeds"]]
        assert bad and all(0.0 <= p["distance"] <= 1.0 for p in bad)


def test_revoked_group_members_counted_as_independent_outputs():
    base = {
        "seed": 11,
        "scale": 1.0,
        "vocab_size": 256,
        "microbatches": [{"tokens": [[1, 2], [3, 4], [5, 6]]}],
    }
    report = run_merge_trial(
        _payload([[1, 2], [3, 4]], threshold=1e-12, base=base)
    )
    r = report["result"]
    assert r["mergeable_count"] == 0 and r["revoked_count"] == 2
    # 全撤销：输出项数不变（成员恢复独立）。
    assert r["output_items_if_independent"] == 256
    assert r["output_items_if_merged"] == 256

    ok_report = run_merge_trial(
        _payload([[1, 2], [3, 4, 5]], threshold=1.0, base=base)
    )
    rr = ok_report["result"]
    # 两个组共 5 个成员合并成 2 项：256 - 5 + 2 = 253。
    assert rr["output_items_if_merged"] == 253


def test_deterministic_results():
    payload = _payload([[1, 2], [3, 4], [5, 6]])
    a = run_merge_trial(payload)
    b = run_merge_trial(payload)

    def distances(report):
        return [
            p["distance"]
            for g in report["result"]["mergeable_groups"]
            + report["result"]["revoked_groups"]
            for p in g["pairs"]
        ]

    assert distances(a) == distances(b)


# ---------------------------------------------------------------------------
# 内存约束：任何投影临时张量的词表轴 <= 128
# ---------------------------------------------------------------------------
def test_chunk_axis_never_exceeds_128(monkeypatch):
    widths = []
    real_range = merge_mod._chunk_ranges

    def spy(vocab_size):
        for start, end in real_range(vocab_size):
            widths.append(end - start)
            yield start, end

    monkeypatch.setattr(merge_mod, "_chunk_ranges", spy)
    big = {
        "seed": 7,
        "scale": 1.0,
        "vocab_size": 4096,
        "microbatches": [{"tokens": [[1, 2], [4095, 4094], [2000, 1000]]}],
        "merge": {"threshold": 0.5, "groups": [[1, 4095], [2000, 2001, 2002]]},
    }
    report = run_merge_trial(big)
    assert widths and max(widths) <= MAX_VOCAB_CHUNK
    assert report["memory"]["max_vocab_chunk"] == MAX_VOCAB_CHUNK
    assert report["memory"]["scores_or_probabilities_materialized_full_vocab"] is False


def test_large_scores_scale_100_remain_finite():
    big = {
        "seed": 424242,
        "scale": 100.0,
        "vocab_size": 4096,
        "microbatches": [
            {"tokens": [[1, 4095], [2048, 0], [4095, 4095], [17, 33]]},
        ],
        "merge": {"threshold": 0.5, "groups": [[1, 17], [2048, 4095]]},
    }
    report = run_merge_trial(big)
    for g in report["result"]["mergeable_groups"] + report["result"]["revoked_groups"]:
        for p in g["pairs"]:
            assert 0.0 <= p["distance"] <= 1.0
