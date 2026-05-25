# 当 RL 学会看地图：从 SA 到 DSA —— 电工杯 B 题的非典型解法

基于强化学习的养老服务站选址优化，使用 **IterableGRPO + Transformer + DSA (DeepSeek Sparse Attention)** 求解组合优化问题。

## 项目概述

电工杯 B 题的核心是养老服务站的选址优化——在 10 个小区里决定在哪建站、建多大，让尽量多的老人被覆盖，同时满意度尽量高。标准解法是完全枚举法，但本项目探索了一条不同的路径：**让模型自己学会选址**。

### 方法论进化树

```
SA → PPO → GRPO(MLP) → GRPO(Transformer) → IterableGRPO(Transformer) → IterableGRPO(Transformer + DSA)
 ↑      ↑         ↑              ↑                    ↑                          ↑
 基线   试错      轻量化         装上眼睛             学会迭代                   擦亮眼睛
```

| 方法 | 覆盖率 | 满意度 | 得分 |
|------|--------|--------|------|
| 完全枚举法 | 93.13% | 0.8777 | 0.8171 |
| SA | 79.7% | 0.8552 | 0.8009 |
| GRPO (MLP) | 53.7% | 0.9362 | 0.7367 |
| GRPO (Transformer) | 93.1% | 0.8742 | 0.8203 |
| **IterableGRPO (Transformer + DSA)** | **90.45%** | **0.8923** | **0.8071** |

### 关键洞察

- **Transformer 自注意力**让策略网络能感知小区之间的空间关系，解决了 MLP "看不见地图" 的问题
- **IterableGRPO** 定期同步参考策略，训练从 500+ 轮骤降到 ~50 轮
- **DSA (DeepSeek Sparse Attention)** 通过动态稀疏化注意力过滤冗余连接，覆盖率进一步提升 1.2 个百分点

## 文件说明

| 文件 | 说明 |
|------|------|
| `two_one.py` | 问题 1 数据处理：人口预测、服务需求计算、距离矩阵构建 |
| `prefab.py` | 数据加载、符号优化模型（SymPy）构建 |
| `floyd.py` | Floyd 最短路径算法 |
| `t1.py` | 方案评估与对比（覆盖率、满意度、站点利用率分析） |
| `t2.py` | 各站点年度利润核算 |
| `t_.py` | MDS 可视化：两种方案覆盖情况对比图 |
| `tt.py` | 价格满意度 S3 评分函数 |
| `test.py` | **主程序**：SA 模拟退火、GRPO 求解器、DSA Transformer 策略网络、完全枚举法 |

## 算法来源说明

- **GRPO (Group Relative Policy Optimization)** 算法实现参考了 [lsdefine/simple_GRPO](https://github.com/lsdefine/simple_GRPO) 以及 GRPO 原论文 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*
- **IterableGRPO** 的迭代参考策略更新机制来源于 GRPO 原论文（DeepSeek-Math）中的设计
- **DSA (DeepSeek Sparse Attention)** 机制复现自 *DeepSeek-V3.2* 论文中的 Lightning Indexer 与 Fine-grained Token Selection
- 所有算法相关代码由 AI 读取论文等参考信息后生成

## 环境依赖

```bash
pip install numpy pandas matplotlib torch gymnasium tqdm scikit-learn openpyxl sympy
```

## 运行方式

```bash
# 主程序：完全枚举 + GRPO (DSA Transformer) 训练与对比
python test.py
```

主要求解器：
- `simulated_annealing()` — 模拟退火基线
- `train_grpo_independent()` — 独立 GRPO 训练（DSA Transformer 策略）
- `cascaded_sa_grpo_satisfaction()` — SA-GRPO 级联优化
- `brute_force_enumeration()` — 完全枚举法（全局最优基准）

## 核心实现

### DSA (DeepSeek Sparse Attention)

```python
# Lightning Indexer: I_{t,s} = Σ w_{t,j} · ReLU(q_{t,j} · k_s)
for j in range(indexer_heads):
    q_idx = indexer_q_proj[j](src)
    k_idx = indexer_k_proj[j](src)
    w_idx = indexer_w_proj[j](src)
    dots = torch.matmul(q_idx, k_idx.transpose(-2, -1))
    I = I + w_idx * F.relu(dots)

# Fine-grained Token Selection: Top-k
top_k_idx = torch.topk(I, k=top_k, dim=-1).indices

# Sparse Multi-Head Attention on selected positions
mask[batch_idx, 0, query_idx, top_k_idx] = 0.0
scores = Q @ K^T / sqrt(d) + mask
attn = softmax(scores)
```

### IterableGRPO

```python
# 每 ref_update_steps 轮同步参考策略
if step_counter % ref_update_steps == 0:
    ref_policy = deepcopy(policy)  # 实时同步最新策略
```

## 许可证

本项目仅用于学习与研究参考。思路为作者原创，代码由 AI 辅助生成，仅供参考。

## 相关链接

- 博客原文：[当 RL 学会看地图：从 SA 到 DSA，一个电工杯 B 题的非典型解法](https://oceroblog.metalstudio.top/%E5%BD%93RL%E5%AD%A6%E4%BC%9A%E7%9C%8B%E5%9C%B0%E5%9B%BE%EF%BC%9A%E4%BB%8ESA%E5%88%B0DSA%EF%BC%8C%E4%B8%80%E4%B8%AA%E7%94%B5%E5%B7%A5%E6%9D%AFB%E9%A2%98%E7%9A%84%E9%9D%9E%E5%85%B8%E5%9E%8B%E8%A7%A3%E6%B3%95.html)
- GRPO 原论文：*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*
- DSA 参考：*DeepSeek-V3.2* 论文
- GRPO 参考实现：[lsdefine/simple_GRPO](https://github.com/lsdefine/simple_GRPO)
