"""词表合并试算（merge trial）：用一步 SGD 后的共享权重 W 比较续写分布。

业务问题
--------
压缩词表前，工程师想把若干词号合并成一个输出项，但要保证这些词分别作为
前缀（输入）时，合并后的续写分布不要差太多。本模块只做**试算**，不真正
改动权重或词表。

计算约定（与任务书逐条对应）
---------------------------
* 复用 :mod:`app.engine` 的解析与分块 SGD：按当前录入的微批做一步
  lr=0.01 的 SGD，取更新后的共享权重 ``W'``（输入嵌入与输出投影仍是同一个
  权重矩阵，weight tying）。
* 对前缀词号 ``p``，续写分数为 ``logits = W'[p] @ W'.T``，概率按**全词表**
  softmax 归一化。分数（logits）与概率都**分块**计算，任何投影临时张量的
  词表轴都不超过 :data:`~app.engine.MAX_VOCAB_CHUNK`（128）；从不物化完整
  的 ``[前缀数, 词表]`` 分数或概率矩阵。
* 评估某个候选组 ``G`` 时，输出项这样构造：``G`` 的成员概率相加记为**一个**
  输出项，其余每个词各占一项（即一次只镜像“当前待合并组”）。
* 组内任意两个成员词 ``a, b`` 分别作前缀时，比较两个镜像分布的**总变差**::

      TV(a,b) = 0.5 * ( |g_a - g_b|
                        + sum_{k not in G} |p_a(k) - p_b(k)| )

  其中 ``g_a = sum_{k in G} p_a(k)`` 是组输出项概率，第二项是组外独立词的
  逐项 L1。组外 L1 逐词表块累加，组项概率只取组成员所在块，全程不物化
  完整的 ``[前缀数, 词表]`` 分数或概率。
* 迭代撤销：若某个待合并组存在超限词对（TV > 阈值），整组撤销、成员恢复成
  独立输出项，再重新检查其余各组，直到没有组再被撤销。
* 由于各组互不重叠，单个组的合并分布不依赖其它组是否存在，词对距离一次
  算清即可；外层仍按任务书实现完整的“撤销—重检”循环。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from .engine import (
    MAX_VOCAB_CHUNK,
    _as_float,
    _as_int,
    _chunk_ranges,
    _require,
    parse_payload,
    run_chunked,
)

# ---------------------------------------------------------------------------
# 输入约束（任务书给定）
# ---------------------------------------------------------------------------
MIN_GROUPS = 1
MAX_GROUPS = 4
MIN_GROUP_MEMBERS = 2
MAX_GROUP_MEMBERS = 8
MIN_TV_THRESHOLD = 0.0
MAX_TV_THRESHOLD = 1.0
# 总变差按定义落在 [0,1]；float64 的分块累加可能有 1e-16 量级的越界噪声。
TV_CLAMP_EPS = 1e-12


@dataclass(frozen=True)
class MergeRequest:
    """解析后的合并试算请求。"""

    threshold: float
    groups: Tuple[Tuple[int, ...], ...]


@dataclass
class PairDistance:
    a: int
    b: int
    distance: float
    exceeds: bool


@dataclass
class GroupReport:
    index: int
    members: List[int]
    pair_count: int
    max_distance: float
    pairs: List[PairDistance] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------
def parse_merge_payload(payload: dict) -> Tuple[object, MergeRequest]:
    """校验整个请求，返回 ``(RunSpec, MergeRequest)``。

    顶层允许字段为引擎既有字段加上 ``merge``；``merge`` 内只允许
    ``groups`` 与 ``threshold``，出现未知字段一律拒绝。
    """
    _require(isinstance(payload, dict), "请求体必须是 JSON 对象")
    engine_allowed = {"seed", "scale", "vocab_size", "microbatches", "tokens"}
    unknown = sorted(set(payload) - engine_allowed - {"merge"})
    _require(not unknown, f"未知字段：{', '.join(unknown)}")

    # 复用引擎的全部微批/词表/种子校验（含未知字段检查，故先剔除 merge）。
    engine_payload = {k: v for k, v in payload.items() if k != "merge"}
    spec = parse_payload(engine_payload)

    merge = payload.get("merge")
    _require(isinstance(merge, dict), "merge 必须是对象，包含 groups 与 threshold")
    merge_unknown = sorted(set(merge) - {"groups", "threshold"})
    _require(not merge_unknown, f"merge 内未知字段：{', '.join(merge_unknown)}")

    threshold = _as_float("总变差阈值 threshold", merge.get("threshold", 1.0))
    _require(
        MIN_TV_THRESHOLD <= threshold <= MAX_TV_THRESHOLD,
        f"总变差阈值 threshold 必须在 [{MIN_TV_THRESHOLD}, {MAX_TV_THRESHOLD}] 内",
    )

    raw_groups = merge.get("groups")
    _require(
        isinstance(raw_groups, list) and MIN_GROUPS <= len(raw_groups) <= MAX_GROUPS,
        f"候选组数必须在 {MIN_GROUPS}～{MAX_GROUPS} 之间",
    )

    groups: List[Tuple[int, ...]] = []
    seen: set[int] = set()
    for gi, raw_group in enumerate(raw_groups):
        label = f"第 {gi + 1} 组"
        _require(
            isinstance(raw_group, (list, tuple)),
            f"{label} 必须是词号数组",
        )
        _require(
            MIN_GROUP_MEMBERS <= len(raw_group) <= MAX_GROUP_MEMBERS,
            f"{label} 的词号数量必须在 {MIN_GROUP_MEMBERS}～{MAX_GROUP_MEMBERS} 之间",
        )
        members: List[int] = []
        local_seen: set[int] = set()
        for raw_id in raw_group:
            word_id = _as_int(f"{label} 词号", raw_id)
            _require(
                0 <= word_id < spec.vocab_size,
                f"{label} 词号 {word_id} 超出词表 [0, {spec.vocab_size})",
            )
            _require(word_id not in local_seen, f"{label} 内词号 {word_id} 重复")
            _require(
                word_id not in seen,
                f"词号 {word_id} 在多个候选组中出现：各组必须互不重叠",
            )
            local_seen.add(word_id)
            seen.add(word_id)
            members.append(word_id)
        groups.append(tuple(members))

    return spec, MergeRequest(threshold=threshold, groups=tuple(groups))


# ---------------------------------------------------------------------------
# 分块数值：前缀归一化概率（全词表 softmax，词表轴始终 <= 128）
# ---------------------------------------------------------------------------
def _prefix_logsumexp_chunked(
    prefix_rows: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """每个前缀在全词表上的 logsumexp，逐块累加（不拼完整分数矩阵）。"""
    lse = torch.full(
        (prefix_rows.shape[0],), -math.inf, dtype=torch.float64
    )
    for start, end in _chunk_ranges(weight.shape[0]):
        logits = prefix_rows @ weight[start:end].t()  # [P, <=128]
        lse = torch.logaddexp(lse, torch.logsumexp(logits, dim=1))
        del logits
    return lse


def _group_member_probabilities(
    prefix_rows: torch.Tensor,
    weight: torch.Tensor,
    members: Sequence[int],
    lse: torch.Tensor,
) -> torch.Tensor:
    """各前缀在组成员词上的归一化概率，返回 ``[P, len(members)]``。

    只对组成员所在的词表块做投影；每块临时张量词表轴 <= 128。
    """
    member_list = list(members)
    out = torch.empty(
        (prefix_rows.shape[0], len(member_list)), dtype=torch.float64
    )
    # 成员按词号排序后可按块顺序扫描。
    order = sorted(range(len(member_list)), key=lambda k: member_list[k])
    for start, end in _chunk_ranges(weight.shape[0]):
        block_ids = [
            k for k in order if start <= member_list[k] < end
        ]
        if not block_ids:
            continue
        logits = prefix_rows @ weight[start:end].t()  # [P, <=128]
        probs = torch.exp(logits - lse.unsqueeze(1))
        for k in block_ids:
            out[:, k] = probs[:, member_list[k] - start]
        del logits, probs
    return out


def _outside_l1_chunked(
    prefix_rows_a: torch.Tensor,
    prefix_rows_b: torch.Tensor,
    weight: torch.Tensor,
    lse_a: torch.Tensor,
    lse_b: torch.Tensor,
    member_mask: torch.Tensor,
) -> float:
    """组外独立词上的 L1 距离 ``sum_{k not in G} |p_a(k) - p_b(k)|``。

    逐词表块投影并立即累加，任何临时张量的词表轴都 <= 128，
    从不保存全词表概率。``member_mask`` 是长度为词表大小的布尔掩码
    （组成员位置为 True），本身不带词表分块轴。
    """
    total = torch.zeros((), dtype=torch.float64)
    for start, end in _chunk_ranges(weight.shape[0]):
        logits_a = prefix_rows_a @ weight[start:end].t()  # [1, <=128]
        logits_b = prefix_rows_b @ weight[start:end].t()
        probs_a = torch.exp(logits_a - lse_a.unsqueeze(1))
        probs_b = torch.exp(logits_b - lse_b.unsqueeze(1))
        outside = ~member_mask[start:end]
        total += (probs_a - probs_b).abs().masked_select(outside).sum()
        del logits_a, logits_b, probs_a, probs_b
    return float(total.item())


def _clamp_tv(value: float) -> float:
    """把浮点噪声导致的轻微越界夹回 [0,1]，超出噪声范围则原样保留。"""
    if -TV_CLAMP_EPS <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + TV_CLAMP_EPS:
        return 1.0
    return value


# ---------------------------------------------------------------------------
# 迭代撤销
# ---------------------------------------------------------------------------
def _evaluate_group(
    weight: torch.Tensor,
    members: Tuple[int, ...],
    threshold: float,
    cache: Dict[Tuple[int, ...], List[PairDistance]],
) -> List[PairDistance]:
    """计算一个候选组内全部词对的总变差距离（带缓存）。

    镜像分布的输出项为：1 个组输出项（组成员概率之和）+ 组外每个词各一项。
    因此词对 (a, b) 的总变差为::

        TV = 0.5 * ( |g_a - g_b|                         # 组输出项之差
                   + sum_{k not in G} |p_a(k) - p_b(k)| )  # 组外独立词逐项差

    组外词的 L1 逐词表块累加，临时张量词表轴始终 <= 128。
    """
    if members in cache:
        return cache[members]

    prefix_idx = torch.tensor(list(members), dtype=torch.long)
    prefix_rows = weight.index_select(0, prefix_idx)       # [m, 16]
    lse = _prefix_logsumexp_chunked(prefix_rows, weight)   # [m]
    member_probs = _group_member_probabilities(
        prefix_rows, weight, members, lse
    )                                                      # [m, m]
    group_item = member_probs.sum(dim=1)                   # [m]

    member_mask = torch.zeros(weight.shape[0], dtype=torch.bool)
    member_mask[torch.tensor(list(members), dtype=torch.long)] = True

    pairs: List[PairDistance] = []
    m = len(members)
    for i in range(m):
        for j in range(i + 1, m):
            outside_l1 = _outside_l1_chunked(
                prefix_rows[i:i + 1],
                prefix_rows[j:j + 1],
                weight,
                lse[i:i + 1],
                lse[j:j + 1],
                member_mask,
            )
            tv = 0.5 * (
                abs(float(group_item[i].item()) - float(group_item[j].item()))
                + outside_l1
            )
            tv = _clamp_tv(tv)
            pairs.append(
                PairDistance(
                    a=int(members[i]),
                    b=int(members[j]),
                    distance=tv,
                    exceeds=tv > threshold,
                )
            )
    cache[members] = pairs
    return pairs


def run_merge_trial(payload: dict) -> dict:
    """合并试算主入口：解析 → 一步 SGD → 分块分布比较 → 迭代撤销。"""
    spec, request = parse_merge_payload(payload)

    # 当前微批上的一步 SGD，取更新后的共享权重 W'。
    chunked = run_chunked(spec)
    weight = chunked.updated_weight

    cache: Dict[Tuple[int, ...], List[PairDistance]] = {}
    active: List[Tuple[int, ...]] = list(request.groups)
    revoked: List[GroupReport] = []
    iterations = 0

    # 撤销—重检循环：每轮检查所有仍待合并的组；只要本轮有组被撤销，
    # 其成员恢复为独立输出项，再重新检查剩余各组，直到一轮内无撤销。
    while active:
        iterations += 1
        revoked_this_round: List[GroupReport] = []
        surviving: List[Tuple[int, ...]] = []
        for gi, members in enumerate(active):
            pairs = _evaluate_group(weight, members, request.threshold, cache)
            offending = [p for p in pairs if p.exceeds]
            if offending:
                revoked_this_round.append(
                    GroupReport(
                        index=gi,
                        members=list(members),
                        pair_count=len(pairs),
                        max_distance=max(p.distance for p in pairs),
                        pairs=pairs,
                    )
                )
            else:
                surviving.append(members)
        if not revoked_this_round:
            break
        revoked.extend(revoked_this_round)
        active = surviving

    # 最终存活组再生成一次完整报告（距离已在缓存中，不重复计算）。
    mergeable: List[GroupReport] = []
    for gi, members in enumerate(active):
        pairs = _evaluate_group(weight, members, request.threshold, cache)
        mergeable.append(
            GroupReport(
                index=gi,
                members=list(members),
                pair_count=len(pairs),
                max_distance=max(p.distance for p in pairs),
                pairs=pairs,
            )
        )

    total_pairs = sum(
        len(members) * (len(members) - 1) // 2 for members in request.groups
    )
    # 只有最终可合并的组才减少输出项：每个存活组把 len(members) 个独立项
    # 压成 1 项；被撤销组的成员恢复为独立项，不参与压缩。
    merged_members = sum(len(g.members) for g in mergeable)
    merged_output_items = spec.vocab_size - merged_members + len(mergeable)

    return {
        "ok": True,
        "message": (
            f"试算完成：{len(mergeable)} 个组可合并，{len(revoked)} 个组被撤销"
        ),
        "config": {
            "vocab_size": spec.vocab_size,
            "embed_dim": weight.shape[1],
            "seed": spec.seed,
            "scale": spec.scale,
            "learning_rate": 0.01,
            "microbatches": len(spec.microbatches),
            "threshold": request.threshold,
            "candidate_groups": len(request.groups),
            "candidate_members": sum(len(g) for g in request.groups),
            "candidate_pairs": total_pairs,
        },
        "sgd": {
            "loss": chunked.loss,
            "valid_tokens": chunked.valid_tokens,
            "token_count": chunked.token_count,
            "updated_weight_norm": float(chunked.updated_weight.norm().item()),
        },
        "result": {
            "iterations": iterations,
            "mergeable_count": len(mergeable),
            "revoked_count": len(revoked),
            # 全部候选组成员保持独立输出时的输出项数；按最终建议合并后的项数。
            "output_items_if_independent": spec.vocab_size,
            "output_items_if_merged": merged_output_items,
            "mergeable_groups": [_group_to_dict(g, request.threshold) for g in mergeable],
            "revoked_groups": [_group_to_dict(g, request.threshold) for g in revoked],
        },
        "memory": {
            "max_vocab_chunk": MAX_VOCAB_CHUNK,
            "scores_or_probabilities_materialized_full_vocab": False,
            "chunked_logsumexp": True,
            "probability_normalization": "full_vocab_softmax",
            "largest_projection_temp_vocab_axis": MAX_VOCAB_CHUNK,
        },
    }


def _group_to_dict(group: GroupReport, threshold: float) -> dict:
    return {
        "candidate_index": group.index,
        "members": group.members,
        "pair_count": group.pair_count,
        "max_distance": group.max_distance,
        "exceeds_threshold": group.max_distance > threshold,
        "pairs": [
            {
                "a": p.a,
                "b": p.b,
                "distance": p.distance,
                "exceeds": p.exceeds,
            }
            for p in group.pairs
        ],
    }
