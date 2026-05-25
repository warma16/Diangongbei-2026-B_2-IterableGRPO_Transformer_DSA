"""
养老服务站选址问题 —— SA + GRPO（Transformer 策略网络 with DeepSeek Sparse Attention）
包含：Floyd 预处理、模拟退火、稀疏注意力 GRPO 求解器、SA-GRPO 级联优化
严格遵循 DeepSeek-V3.2-Exp 论文中的 DSA 机制
"""
TOTAL_BUDGET = 140
import numpy as np
import pandas as pd
import random
import time
import heapq
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

DISALLOW_CAS = True

# 导入数据（请确保 two_one.py 在同一目录下）
from two_one import (
    exp_dist_mat,           # 原始距离矩阵
    P,                      # 各小区老人总数
    D,                      # 各小区日均总需求
    building_costs_,        # 建设成本 [18, 32, 45]
    capacities,             # 日服务容量 [1000, 2000, 3000]
    can_be_select_M,        # 可达性矩阵（bool）
    dist_mat,               # 原始距离矩阵
    total_P                 # 总老人数
)

# ================= Floyd 预处理 =================
def floyd(graph):
    n = len(graph)
    dist = graph.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

dist_mat_opt = floyd(exp_dist_mat)
can_serve = (dist_mat_opt <= 1000)   # 最短路径可达矩阵

# ================= 满意度与评估函数 =================
WEIGHTS = (0.2, 0.3, 0.5)   # S1, S2, S3 权重

def satisfaction(dist_km, utilization):
    if dist_km <= 300:
        s1 = 1.00
    elif dist_km <= 500:
        s1 = 0.90
    elif dist_km <= 650:
        s1 = 0.75
    else:
        s1 = 0.60

    if utilization <= 0.60:
        s2 = 1.00
    elif utilization <= 0.75:
        s2 = 0.93
    elif utilization <= 0.85:
        s2 = 0.85
    elif utilization <= 0.95:
        s2 = 0.72
    else:
        s2 = 0.60

    s3 = 1.00   # 基准价
    return WEIGHTS[0] * s1 + WEIGHTS[1] * s2 + WEIGHTS[2] * s3

def evaluate_solution(solution):
    total_cost = 0
    station_info = []
    for i in range(10):
        s = solution[i]
        if s > 0:
            total_cost += building_costs_[s - 1]
            station_info.append({
                'idx': i,
                'cap': capacities[s - 1],
                'remain': capacities[s - 1],
                'demand': 0.0
            })
    if total_cost > TOTAL_BUDGET:
        return -np.inf, (0, 0)

    assign = {}
    covered = 0
    for k in range(10):
        if P[k] == 0: continue
        candidates = []
        for st in station_info:
            i = st['idx']
            if can_serve[i][k]:
                sat = satisfaction(dist_mat_opt[i][k], 0)
                candidates.append((sat, st))
        if not candidates: continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, st in candidates:
            if st['remain'] >= D[k]:
                st['remain'] -= D[k]
                st['demand'] += D[k]
                assign[k] = st
                covered += P[k]
                break

    coverage = covered / total_P if total_P > 0 else 0
    total_sat = 0.0
    for k, st in assign.items():
        i = st['idx']
        d = dist_mat_opt[i][k]
        utilization = st['demand'] / st['cap'] if st['cap'] > 0 else 0
        sat_score = satisfaction(d, utilization)
        total_sat += P[k] * sat_score
    avg_sat = total_sat / covered if covered > 0 else 0
    score = coverage * avg_sat
    return score, (coverage, avg_sat)

# ================= 模拟退火求解器 =================
def random_feasible_solution():
    while True:
        sol = [0] * 10
        budget = TOTAL_BUDGET
        order = list(range(10))
        random.shuffle(order)
        for i in order:
            possible = [s for s in [1, 2, 3] if building_costs_[s - 1] <= budget]
            if possible:
                s = random.choice(possible)
                sol[i] = s
                budget -= building_costs_[s - 1]
        score, _ = evaluate_solution(sol)
        if score > -np.inf:
            return sol

def mutate(solution):
    new_sol = solution.copy()
    op = random.random()
    if op < 0.6:
        i = random.randint(0, 9)
        if new_sol[i] == 0:
            possible = [s for s in [1, 2, 3] 
                       if building_costs_[s - 1] <= TOTAL_BUDGET - sum(building_costs_[x - 1] for x in new_sol if x > 0)]
            if possible:
                new_sol[i] = random.choice(possible)
        else:
            new_sol[i] = 0
    elif op < 0.9:
        built = [i for i in range(10) if new_sol[i] > 0]
        if built:
            i = random.choice(built)
            cur = new_sol[i]
            choices = [x for x in [1, 2, 3] if x != cur]
            if choices:
                current_cost = sum(building_costs_[x - 1] for x in new_sol if x > 0)
                for c in choices:
                    if current_cost - building_costs_[cur - 1] + building_costs_[c - 1] <= TOTAL_BUDGET:
                        new_sol[i] = c
                        break
    else:
        i, j = random.sample(range(10), 2)
        new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
    return new_sol

def simulated_annealing(T0=500, T_min=0.01, alpha=0.95, max_stall=200, verbose=True):
    sol = random_feasible_solution()
    score, (coverage, avg_sat) = evaluate_solution(sol)
    best_sol = sol.copy()
    best_score = score
    best_coverage = coverage
    best_avg_sat = avg_sat
    T = T0
    stall = 0
    iteration = 0
    while T > T_min and stall < max_stall:
        new_sol = mutate(sol)
        new_score, (new_coverage, new_avg_sat) = evaluate_solution(new_sol)
        if new_score > -np.inf:
            delta = new_score - score
            if delta > 0 or random.random() < np.exp(delta / T):
                sol = new_sol
                score = new_score
                coverage = new_coverage
                avg_sat = new_avg_sat
                if new_score > best_score:
                    best_sol = new_sol.copy()
                    best_score = new_score
                    best_coverage = new_coverage
                    best_avg_sat = new_avg_sat
                    stall = 0
                else:
                    stall += 1
            else:
                stall += 1
        else:
            stall += 1
        T *= alpha
        iteration += 1
    if verbose:
        print(f"SA 完成。共 {iteration} 轮，最优得分: {best_score:.4f}")
    return best_sol, best_score, [], best_coverage, best_avg_sat

# ================= DeepSeek Sparse Attention (DSA) 核心实现 =================
class DSAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA)
    严格遵循论文公式:
        I_{t,s} = Σ_{j=1}^{H^l} w_{t,j}^l · ReLU(q_{t,j}^l · k_s^l)
    然后对每个 query 选取 Top‑k 的 key‑value 对，仅在这些位置上计算标准注意力。
    """
    def __init__(self, d_model, nhead, indexer_heads=4, top_k=3, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.indexer_heads = indexer_heads
        self.top_k = top_k

        # 标准注意力投影 (Q, K, V)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Lightning indexer 组件
        # 每个 indexer head 有自己的 q 和 k 投影，输出维度为 head_dim_index (这里简化为 head_dim)
        # 论文中的 d^l 可小于 d_model，但为了保持能力，我们使用 head_dim
        self.indexer_q_proj = nn.ModuleList([
            nn.Linear(d_model, self.head_dim) for _ in range(indexer_heads)
        ])
        self.indexer_k_proj = nn.ModuleList([
            nn.Linear(d_model, self.head_dim) for _ in range(indexer_heads)
        ])
        # 权重 w_{t,j}^l 从 query token 通过线性层得到标量
        self.indexer_w_proj = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(indexer_heads)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query, key, value: [batch, seq_len, d_model]
            attn_mask: 可选，但 DSA 不需要额外的 mask，由 top‑k 隐式决定
            key_padding_mask: [batch, seq_len] bool, True 表示 padding 位置
        Returns:
            out: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = query.shape

        # ----- 1. 标准 QKV 投影并拆分为多头 -----
        Q = self.q_proj(query).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # [b, nhead, seq, head_dim]
        K = self.k_proj(key).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # ----- 2. Lightning indexer 计算 I_{t,s} -----
        # 对每个 query token t 和每个 key token s，计算 I[t,s]
        # 形状: [batch, seq_len, seq_len]
        I = torch.zeros(batch, seq_len, seq_len, device=query.device)

        for j in range(self.indexer_heads):
            # 当前 indexer head 的 q, k, w
            q_idx = self.indexer_q_proj[j](query)  # [b, seq, head_dim]
            k_idx = self.indexer_k_proj[j](key)    # [b, seq, head_dim]
            w_idx = self.indexer_w_proj[j](query)  # [b, seq, 1]

            # 计算点积: q_idx * k_idx^T -> [b, seq, seq]
            dots = torch.matmul(q_idx, k_idx.transpose(-2, -1))  # [b, seq, seq]
            relu_dots = F.relu(dots)
            # w 扩展为 [b, seq, 1] 与 relu_dots 广播相乘并累加
            I = I + w_idx * relu_dots   # w_idx 形状 [b, seq, 1] 自动广播

        # 可选：对 padding 位置设极小值 (如果 key_padding_mask 提供)
        if key_padding_mask is not None:
            # key_padding_mask: [b, seq] True 为 padding
            I = I.masked_fill(key_padding_mask.unsqueeze(1), -1e9)

        # ----- 3. Fine-grained token selection: 对每个 query 选取 top‑k key 索引 -----
        top_k_val = min(self.top_k, seq_len)
        top_k_indices = torch.topk(I, k=top_k_val, dim=-1).indices  # [b, seq, top_k]

        # ----- 4. 构建稀疏注意力权重 (直接计算全 QK^T 并 mask) -----
        # 计算标准注意力分数: Q * K^T / sqrt(d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [b, nhead, seq, seq]
        # 平均多头分数，得到 [b, seq, seq]
        scores = scores.mean(dim=1)
        
        # 构建 mask：非 top-k 位置设为 -inf
        mask = torch.full_like(scores, -1e9)
        # 批量填充选中位置为 0
        batch_indices = torch.arange(batch).view(batch, 1, 1)  # [b, 1, 1]
        query_indices = torch.arange(seq_len).view(1, seq_len, 1)  # [1, seq, 1]
        mask[batch_indices, query_indices, top_k_indices] = 0.0
        
        scores = scores + mask
        # softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)  # [b, seq, seq]
        attn_weights = self.dropout(attn_weights)

        # ----- 5. 计算注意力输出 (用原始 V) -----
        # V 形状 [b, nhead, seq, head_dim]，先平均多头得到 [b, seq, head_dim]
        V_avg = V.mean(dim=1)  # [b, seq, head_dim]
        out = torch.matmul(attn_weights, V_avg)  # [b, seq, head_dim]
        # 投影回 d_model
        out = self.out_proj(out.view(batch, seq_len, -1))
        return out, attn_weights

class SparseTransformerEncoderLayer(nn.Module):
    """
    完整的 DeepSeek Sparse Attention (DSA) 编码器层
    内部实现 Lightning Indexer、Top‑k 选择、稀疏多头注意力、FFN。
    """
    def __init__(self, d_model, nhead, indexer_heads=4, top_k=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.indexer_heads = indexer_heads
        self.top_k = top_k

        # 标准注意力投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Lightning Indexer 投影
        self.indexer_q_proj = nn.ModuleList([
            nn.Linear(d_model, self.head_dim) for _ in range(indexer_heads)
        ])
        self.indexer_k_proj = nn.ModuleList([
            nn.Linear(d_model, self.head_dim) for _ in range(indexer_heads)
        ])
        self.indexer_w_proj = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(indexer_heads)
        ])

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化与 dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [batch, seq_len, d_model]
            src_mask: 暂未使用（DSA 自身选择 Top‑k）
            src_key_padding_mask: [batch, seq_len] bool, True 为 padding 位置
        Returns:
            out: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = src.shape

        # 1. QKV 投影并分成多头
        Q = self.q_proj(src).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)   # [b, nhead, seq, hd]
        K = self.k_proj(src).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(src).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # 2. Lightning Indexer: 计算重要性得分 I_{t,s}
        I = torch.zeros(batch, seq_len, seq_len, device=src.device)
        for j in range(self.indexer_heads):
            q_idx = self.indexer_q_proj[j](src)      # [b, seq, hd]
            k_idx = self.indexer_k_proj[j](src)      # [b, seq, hd]
            w_idx = self.indexer_w_proj[j](src)      # [b, seq, 1]
            dots = torch.matmul(q_idx, k_idx.transpose(-2, -1))   # [b, seq, seq]
            relu_dots = F.relu(dots)
            I = I + w_idx * relu_dots                # 加权累加

        if src_key_padding_mask is not None:
            # 将 padding 位置设为极小值
            I = I.masked_fill(src_key_padding_mask.unsqueeze(1), -1e9)

        # 3. Top‑k 选择
        top_k_val = min(self.top_k, seq_len)
        top_k_idx = torch.topk(I, k=top_k_val, dim=-1).indices   # [b, seq, top_k]

        # 4. 构建稀疏 mask (形状 [b, 1, seq, seq]，便于与多头分数相加)
        mask = torch.full((batch, 1, seq_len, seq_len), -1e9, device=src.device)
        # 向量化填充选中位置为 0
        batch_idx = torch.arange(batch).view(batch, 1, 1)        # [b, 1, 1]
        query_idx = torch.arange(seq_len).view(1, seq_len, 1)    # [1, seq, 1]
        mask[batch_idx, 0, query_idx, top_k_idx] = 0.0

        # 5. 稀疏多头注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)   # [b, nhead, seq, seq]
        scores = scores + mask                     # 非 Top‑k 位置变为 -inf
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 6. 注意力输出
        attn_out = torch.matmul(attn, V)           # [b, nhead, seq, hd]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        # 7. 残差 & 归一化
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # 8. 前馈网络
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)

        return src

class SparseTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, indexer_heads=4, top_k=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(d_model, nhead, indexer_heads, top_k, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output

# ================= Transformer 策略网络（使用 DSA） =================
class TransformerPolicy(nn.Module):
    """
    基于 DeepSeek Sparse Attention 的策略网络，输入历史观测序列，输出当前步动作 logits
    """
    def __init__(self, obs_dim, d_model=128, nhead=4, num_layers=2, max_seq_len=10,
                 indexer_heads=4, top_k=3):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(obs_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.01)
        # 使用自定义的稀疏 Transformer 编码器
        self.transformer = SparseTransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            indexer_heads=indexer_heads,
            top_k=top_k,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        self.action_head = nn.Linear(d_model, 4)  # 4 个动作（0,1,2,3）

    def forward(self, obs_seq):
        """
        obs_seq: tensor of shape [batch, seq_len, obs_dim]
        返回: logits of shape [batch, 4]
        """
        batch, seq_len, _ = obs_seq.shape
        x = self.embed(obs_seq)  # [batch, seq_len, d_model]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.transformer(x)  # [batch, seq_len, d_model]
        last_out = x[:, -1, :]   # [batch, d_model]
        logits = self.action_head(last_out)
        return logits

# ================= GRPO 环境（支持 Transformer 需要历史观测） =================
class ElderlyServiceSeqEnv(gym.Env):
    """环境接口保持不变，注意外部需要累积历史观测"""
    def __init__(self, P, D, can_serve, building_costs, capacities, dist_mat, evaluate_fn,
                 reference_solution=None, lock_coverage=False):
        super().__init__()
        self.P = P
        self.D = D
        self.can_serve = can_serve
        self.building_costs = building_costs
        self.capacities = capacities
        self.dist_mat = dist_mat
        self.evaluate_fn = evaluate_fn
        self.n_communities = len(P)
        self.total_budget = TOTAL_BUDGET

        self.lock_coverage = lock_coverage
        self.reference_solution = reference_solution
        if self.reference_solution is not None:
            self.has_reference = True
            self.ref_actions = np.array(self.reference_solution, dtype=int)
            _, (self.ref_cov, _) = self.evaluate_fn(list(self.ref_actions))
        else:
            self.has_reference = False
            self.ref_actions = np.zeros(self.n_communities, dtype=int)
            self.ref_cov = 0.0

        self.action_space = spaces.Discrete(4)
        base_dim = 1 + 1 + self.n_communities + self.n_communities + self.n_communities
        ref_dim = self.n_communities if self.has_reference else 0
        self.obs_dim = base_dim + ref_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def _get_obs(self):
        base_obs = np.concatenate([
            np.array([self.current_idx / self.n_communities], dtype=np.float32),
            np.array([self.budget_remaining / self.total_budget], dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            self.P.astype(np.float32),
            self.D.astype(np.float32),
        ])
        if self.has_reference:
            ref_obs = self.ref_actions.astype(np.float32)
            return np.concatenate([base_obs, ref_obs]).astype(np.float32)
        else:
            return base_obs.astype(np.float32)

    def action_masks(self):
        masks = np.ones(4, dtype=bool)
        try:
            if self.lock_coverage and self.has_reference:
                ref_action = self.ref_actions[self.current_idx]
                if ref_action == 0:
                    masks[1] = masks[2] = masks[3] = False
                else:
                    masks[0] = False
                    for a in range(1, 4):
                        if a < ref_action:
                            masks[a] = False
            else:
                for a in range(4):
                    cost = self.building_costs[a-1] if a > 0 else 0
                    if cost > self.budget_remaining + 1e-6:
                        masks[a] = False
        except:
            pass
        if not np.any(masks):
            masks[0] = True
        return masks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        self.budget_remaining = self.total_budget
        self.actions = np.zeros(self.n_communities, dtype=int)
        return self._get_obs(), {}

    def step(self, action):
        self.actions[self.current_idx] = action
        cost = self.building_costs[action-1] if action > 0 else 0
        self.budget_remaining -= cost
        self.current_idx += 1
        done = (self.current_idx >= self.n_communities)
        if done:
            total_cost = sum(self.building_costs[a-1] for a in self.actions if a > 0)
            if total_cost > self.total_budget:
                reward = -1.0
            else:
                score, _ = self.evaluate_fn(list(self.actions))
                reward = score if score > -np.inf else -1.0
        else:
            reward = 0.0
        return self._get_obs(), reward, done, False, {}

# ================= GRPO 求解器（使用 DSA Transformer 策略） =================
class IterativeGRPOSolver:
    def __init__(self, env, group_size=8, lr=3e-4, clip_eps=0.2, beta=0.04,
                 ref_update_steps=50, replay_ratio=0.1, reward_model_lr=1e-3):
        self.env = env
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.beta = beta
        self.ref_update_steps = ref_update_steps
        self.replay_ratio = replay_ratio

        obs_dim = env.observation_space.shape[0]
        # 策略网络：使用 DSA Transformer
        self.policy = TransformerPolicy(obs_dim, d_model=128, nhead=4, num_layers=2, max_seq_len=10,
                                        indexer_heads=4, top_k=3)
        self.ref_policy = deepcopy(self.policy)
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # 奖励模型（仍为 MLP，可保留）
        self.reward_model = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=reward_model_lr)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.step_counter = 0
        self.replay_buffer = []
        self.best_score = -np.inf
        self.best_solution = None

    def _get_masked_logits(self, obs_seq, action_masks, use_ref=False):
        net = self.ref_policy if use_ref else self.policy
        obs_tensor = torch.FloatTensor(np.stack(obs_seq, axis=0)).unsqueeze(0)
        logits = net(obs_tensor)
        masked = logits.clone().squeeze(0)
        masked[~torch.BoolTensor(action_masks)] = -1e9
        return masked

    def sample_trajectory(self):
        obs, _ = self.env.reset()
        done = False
        states, actions, old_log_probs, masks = [], [], [], []
        obs_history = []
        while not done:
            action_masks = self.env.action_masks()
            obs_history.append(obs)
            logits = self._get_masked_logits(obs_history, action_masks)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            states.append(obs)
            actions.append(action.item())
            old_log_probs.append(dist.log_prob(action))
            masks.append(action_masks)
            obs, reward, done, _, _ = self.env.step(action.item())
        score, _ = self.env.evaluate_fn(list(self.env.actions))
        return states, actions, old_log_probs, masks, max(score, 0.0)

    def compute_group_advantages(self, rewards):
        rewards = np.array(rewards)
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    def train_reward_model(self, epochs=5):
        if len(self.replay_buffer) < self.group_size:
            return
        self.reward_model.train()
        for _ in range(epochs):
            batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), 64))
            states_batch = torch.FloatTensor([item[0] for item in batch])
            labels = torch.FloatTensor([item[1] for item in batch]).unsqueeze(1)
            pred = self.reward_model(states_batch)
            loss = nn.MSELoss()(pred, labels)
            self.reward_optimizer.zero_grad()
            loss.backward()
            self.reward_optimizer.step()

    def add_to_replay(self, states, reward):
        avg_state = np.mean(states, axis=0)
        self.replay_buffer.append((avg_state, reward))

    def train_step(self):
        group_data = []
        for _ in range(self.group_size):
            states, actions, old_log_probs, masks, reward = self.sample_trajectory()
            group_data.append({
                'states': states,
                'actions': actions,
                'old_log_probs': old_log_probs,
                'masks': masks,
                'reward': reward
            })

        rewards = [d['reward'] for d in group_data]
        advantages = self.compute_group_advantages(rewards)

        for data in group_data:
            self.add_to_replay(data['states'], data['reward'])

        total_loss = torch.tensor(0.0, requires_grad=True)
        for data, adv in zip(group_data, advantages):
            seq_len = len(data['states'])
            seq_loss = torch.tensor(0.0, requires_grad=True)

            for t in range(seq_len):
                obs_history = data['states'][:t+1]
                action = data['actions'][t]
                old_log_prob = data['old_log_probs'][t]
                action_mask = data['masks'][t]

                logits = self._get_masked_logits(obs_history, action_mask)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_prob = dist.log_prob(torch.tensor(action))

                ref_logits = self._get_masked_logits(obs_history, action_mask, use_ref=True)
                ref_probs = torch.softmax(ref_logits, dim=-1)
                ref_dist = torch.distributions.Categorical(ref_probs)
                ref_log_prob = ref_dist.log_prob(torch.tensor(action))

                ratio = torch.exp(new_log_prob - old_log_prob.detach())
                adv_tensor = torch.tensor(adv, dtype=torch.float32)
                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_tensor
                policy_loss = -torch.min(surr1, surr2)

                kl = torch.exp(ref_log_prob - new_log_prob) - (ref_log_prob - new_log_prob) - 1.0
                token_loss = policy_loss + self.beta * kl
                seq_loss = seq_loss + token_loss

            seq_loss = seq_loss / seq_len
            total_loss = total_loss + seq_loss

        loss = total_loss / self.group_size
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        max_reward = max(rewards)
        if max_reward > self.best_score:
            self.best_score = max_reward
            best_idx = rewards.index(max_reward)
            self.best_solution = group_data[best_idx]['actions']

        self.step_counter += 1
        if self.step_counter % self.ref_update_steps == 0:
            self.update_reference_model()
            self.train_reward_model()

        return loss.item(), max_reward, self.best_score

    def update_reference_model(self):
        self.ref_policy = deepcopy(self.policy)
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        if len(self.replay_buffer) > 0:
            keep_size = max(1, int(len(self.replay_buffer) * self.replay_ratio))
            self.replay_buffer = random.sample(self.replay_buffer, keep_size)

# ================= SA-GRPO 级联优化 (使用 DSA) =================
def cascaded_sa_grpo_satisfaction():
    print("\n" + "=" * 60)
    print("SA-GRPO 级联优化（DSA Transformer 策略，满意度专攻模式）")
    print("=" * 60)

    # 1. SA 底座
    print("\n>>> 第一步：SA 高覆盖率底座 <<<")
    sa_sol, sa_score, _, sa_cov, sa_sat = simulated_annealing(verbose=True)
    print(f"SA 底座方案: {[(i+1, s) for i, s in enumerate(sa_sol) if s > 0]}")
    print(f"SA 覆盖率: {sa_cov:.4f}，满意度: {sa_sat:.4f}，得分: {sa_score:.4f}")

    # 2. 创建环境（可选锁定覆盖率）
    if not DISALLOW_CAS:
        print("\n>>> 第二步：GRPO（DSA Transformer）满意度优化（锁定覆盖率） <<<")
        grpo_env = ElderlyServiceSeqEnv(
            P, D, can_serve, building_costs_, capacities, dist_mat_opt,
            evaluate_solution,
            reference_solution=sa_sol,
            lock_coverage=True
        )
    else:
        print("\n>>> 第二步：GRPO（DSA Transformer）独立 <<<")
        grpo_env = ElderlyServiceSeqEnv(
            P, D, can_serve, building_costs_, capacities, dist_mat_opt,
            evaluate_solution
        )

    # 3. 训练 GRPO（迭代式）
    grpo_solver = IterativeGRPOSolver(grpo_env, group_size=8, lr=3e-4, clip_eps=0.2, beta=0.04)
    n_iterations = 500
    best_score = sa_score
    best_sol = sa_sol.copy()
    best_sat = sa_sat
    best_cov = sa_cov
    history = {'sat': [], 'cov': [], 'score': []}

    with tqdm(total=n_iterations, desc="GRPO（DSA Transformer）满意度优化", unit="iter") as pbar:
        for i in range(n_iterations):
            loss, max_r, best_r = grpo_solver.train_step()
            if grpo_solver.best_solution is not None:
                score, (cov, sat) = evaluate_solution(grpo_solver.best_solution)
                history['sat'].append(sat)
                history['cov'].append(cov)
                history['score'].append(score)
                if score > best_score and cov >= sa_cov * 0.95:
                    best_score = score
                    best_sol = grpo_solver.best_solution.copy()
                    best_sat = sat
                    best_cov = cov
            else:
                history['sat'].append(sa_sat)
                history['cov'].append(sa_cov)
                history['score'].append(sa_score)
            pbar.set_postfix({'best_score': f"{best_score:.4f}", 'sat': f"{best_sat:.4f}", 'cov': f"{best_cov:.4f}"})
            pbar.update(1)

    # 4. 结果对比
    print("\n>>> 优化结果对比 <<<")
    print(f"{'方案':<12} {'覆盖率':<10} {'满意度':<10} {'得分':<10}")
    print(f"{'SA底座':<12} {sa_cov:<10.4f} {sa_sat:<10.4f} {sa_score:<10.4f}")
    print(f"{'级联优化':<12} {best_cov:<10.4f} {best_sat:<10.4f} {best_score:<10.4f}")
    print(f"{'提升':<12} {best_cov-sa_cov:<+10.4f} {best_sat-sa_sat:<+10.4f} {best_score-sa_score:<+10.4f}")

    # 5. 画图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['sat'], 'r-', linewidth=1.5)
    axes[0].axhline(y=sa_sat, color='b', linestyle='--', label='SA Baseline')
    axes[0].set_title('Satisfaction Optimization (DSA Transformer)')
    axes[0].legend()
    axes[1].plot(history['cov'], 'g-', linewidth=1.5)
    axes[1].axhline(y=sa_cov, color='b', linestyle='--', label='SA Baseline')
    axes[1].set_title('Coverage Preservation')
    axes[1].legend()
    axes[2].plot(history['score'], 'b-', linewidth=1.5)
    axes[2].axhline(y=sa_score, color='b', linestyle='--', label='SA Baseline')
    axes[2].set_title('Composite Score')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('sa_grpo_dsa_transformer_optimization.png', dpi=150)
    print("\n训练曲线已保存为 sa_grpo_dsa_transformer_optimization.png")
    return sa_sol, sa_score, best_sol, best_score, history

# ================= 独立 GRPO 训练（DSA Transformer） =================
def train_grpo_independent(n_iterations=500, patience=50):
    """独立 GRPO（DSA Transformer 策略）训练，支持早停"""
    print("\n" + "=" * 60)
    print("GRPO（DSA Transformer 策略）独立求解器")
    print("=" * 60)

    env = ElderlyServiceSeqEnv(
        P, D, can_serve, building_costs_, capacities, dist_mat_opt,
        evaluate_solution
    )

    grpo_solver = IterativeGRPOSolver(env, group_size=10, lr=1e-4, clip_eps=0.2, beta=0.04)

    best_score = -np.inf
    best_sol = None
    history = {'loss': [], 'score': [], 'cov': [], 'sat': []}

    steps_without_improvement = 0
    best_iteration = 0

    with tqdm(total=n_iterations, desc="GRPO（DSA Transformer）训练", unit="iter") as pbar:
        for i in range(n_iterations):
            loss, max_r, best_r = grpo_solver.train_step()
            history['loss'].append(loss)

            if grpo_solver.best_solution is not None:
                score, (cov, sat) = evaluate_solution(grpo_solver.best_solution)
                history['score'].append(score)
                history['cov'].append(cov)
                history['sat'].append(sat)

                if score > best_score:
                    best_score = score
                    best_sol = grpo_solver.best_solution.copy()
                    steps_without_improvement = 0
                    best_iteration = i
                else:
                    steps_without_improvement += 1
            else:
                steps_without_improvement += 1

            if steps_without_improvement >= patience:
                print(f"\n提前停止：连续 {patience} 轮无提升，在第 {i+1} 轮终止。")
                break

            pbar.set_postfix({'best': f"{best_score:.4f}", 'wait': steps_without_improvement})
            pbar.update(1)

    if best_sol is not None:
        b_s, (final_cov, final_sat) = evaluate_solution(best_sol)
        print(f"\nGRPO（DSA Transformer）最优方案: {[(i+1, s) for i, s in enumerate(best_sol) if s > 0]}")
        print(f"得分: {b_s:.4f}，覆盖率: {final_cov:.4f}，满意度: {final_sat:.4f}")
    else:
        print("未找到可行方案。")
        return None, -np.inf, history

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history['score'], 'b-', linewidth=1.5)
    axes[0].set_title('GRPO (DSA Transformer) Best Score')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Score')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['cov'], 'g-', linewidth=1.5, label='Coverage')
    axes[1].plot(history['sat'], 'r-', linewidth=1.5, label='Satisfaction')
    axes[1].set_title('Coverage & Satisfaction')
    axes[1].set_xlabel('Iteration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grpo_dsa_transformer_training.png', dpi=150)
    print("训练曲线已保存为 grpo_dsa_transformer_training.png")

    return best_sol, best_score, history

# ================= 完全枚举法 =================
def brute_force_enumeration():
    """
    遍历所有 4^10 种方案，返回全局最优解。
    """
    print("开始完全枚举（共 4^10 = 1,048,576 种方案）...")
    start_time = time.time()

    best_sol = None
    best_coverage = 0
    best_sat = 0
    best_obj = 0  # coverage × satisfaction（沿用示例论文的目标函数）

    count = 0
    feasible_count = 0

    for sol_tuple in itertools.product([0, 1, 2, 3], repeat=10):
        sol = list(sol_tuple)
        count += 1

        total_cost = sum(building_costs_[s - 1] for s in sol if s > 0)
        if total_cost > TOTAL_BUDGET:
            continue

        feasible_count += 1
        score, (cov, sat) = evaluate_solution(sol)

        if cov == 0 and sat == 0:
            continue

        obj = score

        if obj > best_obj:
            best_obj = obj
            best_coverage = cov
            best_sat = sat
            best_sol = sol[:]

        if feasible_count % 100000 == 0:
            elapsed = time.time() - start_time
            print(f"  已检查 {count:,} 组合, 可行 {feasible_count:,}, "
                  f"当前最优 obj={best_obj:.4f}, 耗时 {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n枚举完成！共 {count:,} 种组合, 可行 {feasible_count:,} 种, 耗时 {elapsed:.1f}秒")

    return best_sol, (best_coverage, best_sat)

# ================= 主程序 =================
if __name__ == "__main__":
    print("===== 完全枚举法（全局最优解）=====")
    enum_sol, (enum_cov, enum_sat) = brute_force_enumeration()
    enum_cost = sum(building_costs_[s - 1] for s in enum_sol if s > 0)
    enum_score = 0.5 * enum_cov + 0.5 * enum_sat
    print(f"枚举法最优方案: {[(i+1, chr(65+i), s) for i, s in enumerate(enum_sol) if s > 0]}")
    print(f"总成本: {enum_cost}万元, 覆盖率: {enum_cov:.4f}, 满意度: {enum_sat:.4f}, 得分: {enum_score:.4f}")

    # 独立 GRPO（DSA Transformer）
    grpo_sol, grpo_score, grpo_hist = train_grpo_independent(500)

    # 对比
    _, (grpo_cov, grpo_sat) = evaluate_solution(grpo_sol)
    print(f"\n===== 方法对比 =====")
    print(f"{'方法':<10} {'得分':<10} {'覆盖率':<10} {'满意度':<10}")
    print(f"{'GRPO(DSA)':<10} {grpo_score:<10.4f} {grpo_cov:<10.4f} {grpo_sat:<10.4f}")

    # 可选：运行级联优化
    # cascaded_sa_grpo_satisfaction()