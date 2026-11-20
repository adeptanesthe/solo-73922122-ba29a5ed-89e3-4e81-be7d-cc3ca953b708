# 共享词嵌入续写训练 · SGD 单步对照校验

面向文本模型训练工程师的**内存节省**实现与**一致性对照**工具：

* 同一个 float64 权重矩阵 `W`（种子 → 标准正态 → ×倍率）既作**输入嵌入**（查表），
  又作**输出投影**（`logits = h · Wᵀ`），即 weight tying。
* 损失为全部**有效**目标（目标号 `-100` 忽略）上的**平均交叉熵**，
  只做**一次** lr = 0.01 的 SGD。
* 自定义 `torch.autograd.Function` 计算损失：
  * 投影临时张量的**词表轴 ≤ 128**（`MAX_VOCAB_CHUNK`）；
  * 前向**不保存**任何分块分数（logits）或概率，只保存共享 W、隐状态、目标号与
    有效行索引；反向按块**重算** logits / logsumexp 再求梯度；
  * 微批损失按各自有效 token 数加权后相加，等价于全局有效 token 平均。
* 独立的**稠密参考**实现不受限（完整 `[tokens, vocab]` logits + `log_softmax`）。
* 网页录入 1–4 个微批（总计 ≤128 token），显示**损失 / 共享 W 梯度 / 更新后权重**
  三项对照误差，全部 ≤1e-8 才显示“通过”；同时验证把同一 token 流重新拆成
  1–4 个微批时损失、梯度、更新后权重不变。
* 训练页另提供**词表合并试算**（`POST /api/merge-trial`）：在一步 SGD 后的
  共享权重 W′ 上，比较候选合并组词分别作前缀时的续写分布总变差，按阈值做
  迭代撤销，给出最终可合并组与每个撤销组的超限词对（详见下文“合并试算”）。
* 内置四个样例：重复 token、不等有效长度（含整批被忽略）、大分数（scale=100，
  词表 4096）、合并试算演示。

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

服务提供：

| 路径 | 说明 |
| --- | --- |
| `GET /` | 录入界面 |
| `GET /healthz` | 健康检查（Compose healthcheck 使用） |
| `GET /api/samples` | 四个内置样例 |
| `POST /api/run` | 运行对照，返回 JSON 报告 |
| `POST /api/merge-trial` | 词表合并试算，返回合并/撤销建议 |

### 本地运行

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> 运行环境为 CPU 版 PyTorch（镜像中显式安装 `--index-url .../whl/cpu`）。
> 本项目用到的都是 float64 CPU 算子，但若本机默认装的是 CUDA 版 PyTorch，
> 数值结果一致。

## 输入格式

```json
{
  "seed": 20240517,
  "scale": 1.0,
  "vocab_size": 256,
  "microbatches": [
    {"name": "dup-A", "tokens": [[4, 4], [4, 4], [7, 4], [4, 7]]},
    {"name": "dup-B", "tokens": [[4, 4], [9, 9], [9, -100], [4, 9]]}
  ]
}
```

* `seed`：0 … 2⁶³−1 的整数；`torch.Generator` 固定种子生成
  `randn(vocab_size, 16, dtype=float64)`。
* `scale`：0.1 … 100（权重倍率）。
* `vocab_size`：256 … 4096；嵌入维度固定 16。
* `microbatches`：1 … 4 个；每项 `tokens` 为 `[输入号, 目标号]` 列表。
  输入号必须落在词表内；目标号为 `-100` 或词表内整数。
* 全部微批的 token 总数（含 `-100` 项）≤ 128。
* 若**所有**目标都是 `-100`，请求返回 422 且不做任何参数更新
  （损失无定义，拒绝运行）。
* 单微批简写：顶层直接给 `"tokens": [...]` 也可。

## 合并试算

压缩词表前，需要判断哪些词可以合并成一个输出项，又不使同组词分别作前缀时
的续写分布差太多。训练页左侧“合并试算”区与 `POST /api/merge-trial` 实现该试算。

### 请求

在 `/api/run` 的同一套微批配置上增加 `merge` 字段：

```json
{
  "seed": 20240517,
  "scale": 1.0,
  "vocab_size": 256,
  "microbatches": [ ... ],
  "merge": {
    "threshold": 0.05,
    "groups": [[4, 7, 9], [10, 11]]
  }
}
```

* `groups`：1～4 个候选组；每组 2～8 个**互不相同**的词号，组与组之间
  **互不重叠**（任一词号只能出现在一个组里），词号必须在词表范围内。
* `threshold`：总变差阈值，0～1 的小数。
* 微批部分的校验规则与 `/api/run` 完全一致（全忽略目标等仍返回 422）。

### 计算过程

1. 用当前微批做一步 lr=0.01 的 SGD（复用引擎的分块自定义 autograd），
   取**更新后的共享权重 W′**；输入嵌入与输出投影仍为同一矩阵。
2. 对前缀词 `p`，续写分数 `logits = W′[p] · W′ᵀ`，概率按**全词表**
   softmax 归一化。
3. 评估候选组 `G` 时，把 `G` 成员的概率**相加记为一个输出项**，其余词各占
   一项（一次只镜像当前待合并组）。组内任意两词 `a, b` 作前缀时，比较两个
   镜像分布的**总变差距离**：

   ```
   TV(a,b) = 0.5 · ( |g_a − g_b| + Σ_{k ∉ G} |p_a(k) − p_b(k)| )
   ```

   其中 `g_a = Σ_{k ∈ G} p_a(k)` 是组输出项概率，第二项是组外独立词的
   逐项 L1。
4. **迭代撤销**：若某待合并组存在超限词对（`TV > threshold`），整组撤销、
   成员恢复成独立输出项，再重新检查剩余各组，直到一轮中没有组再被撤销。
   由于各组互不重叠，单个组的镜像分布不依赖其它组，词对距离只计算一次并
   缓存；外层仍完整执行“撤销—重检”循环，报告给出重检轮次。
5. 分数与概率全程**分块**处理：logsumexp 用 `logaddexp` 逐块累加，组外 L1
   逐块投影后立即归约，任何投影临时张量的**词表轴 ≤ 128**，从不物化完整的
   `[前缀数, 词表]` 分数或概率矩阵。

### 响应（节选）

```jsonc
{
  "ok": true,
  "config": { "threshold": 0.05, "candidate_groups": 2, "candidate_pairs": 4 },
  "sgd": { "loss": 8.73, "valid_tokens": 7, "updated_weight_norm": 63.06 },
  "result": {
    "iterations": 2,
    "mergeable_count": 1,
    "revoked_count": 1,
    "output_items_if_independent": 256,
    "output_items_if_merged": 255,
    "mergeable_groups": [
      { "candidate_index": 0, "members": [10, 11], "max_distance": 0.0287,
        "pairs": [ { "a": 10, "b": 11, "distance": 0.0287, "exceeds": false } ] }
    ],
    "revoked_groups": [
      { "candidate_index": 0, "members": [4, 7, 9], "max_distance": 0.0506,
        "pairs": [
          { "a": 4, "b": 7, "distance": 0.0506, "exceeds": true },
          { "a": 4, "b": 9, "distance": 0.0506, "exceeds": true },
          { "a": 7, "b": 9, "distance": 0.0003, "exceeds": false }
        ] }
    ]
  },
  "memory": { "max_vocab_chunk": 128,
              "scores_or_probabilities_materialized_full_vocab": false,
              "probability_normalization": "full_vocab_softmax" }
}
```

* 最终可合并的组列在 `mergeable_groups`；每个撤销组在 `revoked_groups` 中
  给出全部词对距离，并标记超限词对。
* `output_items_if_merged` 只统计最终存活组的压缩（撤销组成员恢复独立，
  不减少输出项）。
* 试算失败（请求被 422 拒绝或网络错误）时，网页会**清除旧的合并建议**，
  不展示过期结果。

## 对照报告

`POST /api/run` 返回（节选）：

```jsonc
{
  "ok": true,
  "tolerance": 1e-8,
  "chunked": { "loss": 25.70, "grad_norm": 6.84, "weight_norm": 63.94 },
  "dense":   { "loss": 25.70, "grad_norm": 6.84, "weight_norm": 63.94 },
  "errors":  { "loss_abs_diff": 3.6e-15, "grad_max_abs_diff": 2.6e-15,
               "weight_max_abs_diff": 2.2e-16 },
  "split":   { "max_loss_diff": 3.6e-15, "max_grad_diff": 2.2e-16,
               "max_weight_diff": 0.0, "runs": [ ... ] },
  "checks":  { "loss": true, "grad": true, "updated_weight": true,
               "split_loss": true, "split_grad": true, "split_weight": true },
  "memory":  { "max_vocab_chunk": 128, "scores_or_probabilities_saved": false,
               "recompute_in_backward": true, ... }
}
```

* 三项主对照：`loss_abs_diff`、`grad_max_abs_diff`、`weight_max_abs_diff`
  （分块实现 vs 稠密参考）。
* 拆分不变量：把同一 token 流（保持顺序）重新均分为 1–4 个微批，各自跑一遍，
  与用户录入的微批结果比较，再取最大值。
* 只有六个 `checks` 全为 true，`ok` 才为 true，网页才显示“通过”。

## 实现要点

```
app/engine.py       校验、W 生成、自定义 autograd、稠密参考、对照报告
app/merge.py        合并试算：解析、分块续写分布、总变差、迭代撤销
app/main.py         FastAPI 路由（/api/run 与 /api/merge-trial）
app/static/index.html  录入与结果界面（无外部依赖）
app/samples.py      四个内置样例（含合并试算样例）
tests/test_engine.py   数值/约束/内存行为测试
tests/test_merge.py    合并试算的数值、校验、撤销与分块约束测试
tests/test_api.py      HTTP 层测试
```

自定义损失的核心：

```python
class ChunkedSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, hidden, target):
        # 1) 逐词表块算 logits = h @ W[chunk].T（词表轴 ≤128），
        #    logaddexp 累积每个 token 的全局 logsumexp；
        # 2) 只按块取目标 token 的 logit；
        # 3) loss = mean(lse - target_logit)；
        # 4) 只保存 weight / hidden / target / valid_idx。
    @staticmethod
    def backward(ctx, grad_output):
        # 用保存的 hidden 与 W 重算每块 logits，再 exp(logits - lse)，
        # 就地减去 one-hot 后按块累加 dW[start:end] += probs.T @ h，
        # dH += probs @ W[start:end]；不缓存前向的分数或概率。
```

* 重复输入号通过 `index_add_` 累加 `dH`，与 autograd 的 `index_select` 语义一致。
* `logaddexp` 分块累积避免“先拼完整分数矩阵”，对 scale=100 的大分数仍稳定。
* 微批按“有效 token 数 / 全局有效 token 数”加权，因此拆分数不改变结果。

## 测试

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install -r requirements.txt      # 运行服务
python3 -m pip install -r requirements-dev.txt  # 运行测试（含 pytest/httpx）
python3 -m pytest tests -q
```

覆盖：种子可复现、权重共享（与手工 tying 图对比）、`-100` 忽略、全忽略拒绝、
边界与超限校验、128 token 上限、微批拆分不变量、重复 token 梯度累加、
有限差分梯度校验、大分数数值精度、`save_for_backward` 不含分数/概率、词表分块 ≤128、
HTTP 路由与样例端到端通过。合并试算另覆盖：候选组/阈值/重叠等输入校验、分块
总变差与无分块稠密暴力实现的逐项一致（误差 < 1e-12）、确认使用更新后 W′、
全词表归一化、阈值 0/1 边界、多轮迭代撤销、撤销组输出项恢复独立、确定性、
词表 4096 下投影词表轴仍 ≤128、scale=100 时距离仍有限且落在 [0,1]。

## 已知限制

* 仅 CPU、float64、单步 SGD；没有优化器状态、学习率调度或梯度裁剪。
* `scale` 在 W 生成时一次性乘上，等价于把整体 logits 放大；不做损失缩放。
* 网页不做并发限流，单次请求为同步计算（≤128 token、词表 ≤4096 时耗时很短）。
* 对照误差阈值固定 1e-8（float64 下实测约 1e-11 量级或更小）。
