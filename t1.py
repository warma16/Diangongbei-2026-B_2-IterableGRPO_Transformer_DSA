import numpy as np
import pandas as pd

# ================= 载入必要数据 =================
from two_one import (exp_dist_mat, P, D, building_costs_, capacities,
                     can_be_select_M, dist_mat, total_P)

# Floyd预处理
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
can_serve = (dist_mat_opt <= 1000)

# 满意度函数
WEIGHTS = (0.2, 0.3, 0.5)
def satisfaction(dist_km, utilization):
    if dist_km <= 300: s1 = 1.00
    elif dist_km <= 500: s1 = 0.90
    elif dist_km <= 650: s1 = 0.75
    else: s1 = 0.60

    if utilization <= 0.60: s2 = 1.00
    elif utilization <= 0.75: s2 = 0.93
    elif utilization <= 0.85: s2 = 0.85
    elif utilization <= 0.95: s2 = 0.72
    else: s2 = 0.60

    s3 = 1.00
    return WEIGHTS[0]*s1 + WEIGHTS[1]*s2 + WEIGHTS[2]*s3

# GRPO最优方案
def exc(solution):

    # 构建站点信息
    station_info = []
    for i in range(10):
        s = solution[i]
        if s > 0:
            station_info.append({
                'idx': i,
                'name': chr(65+i),
                'scale': s,
                'cap': capacities[s-1],
                'remain': capacities[s-1],
                'demand': 0.0
            })

    # 贪心分配（同evaluate_solution）
    assign = {}  # k -> station_info dict
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

    # 计算每个站点的服务人数、利用率、平均满意度
    print("站点\t覆盖小区\t服务人数\t利用率\t平均满意度")
    for st in station_info:
        # 找出分配给该站点的所有小区
        served_communities = []
        served_pop = 0
        total_sat_weighted = 0.0
        for k, ast in assign.items():
            if ast is st:
                served_communities.append(chr(65+k))
                served_pop += P[k]
                i = st['idx']
                d = dist_mat_opt[i][k]
                util = st['demand'] / st['cap'] if st['cap'] > 0 else 0
                sat_val = satisfaction(d, util)
                total_sat_weighted += P[k] * sat_val

        avg_sat = total_sat_weighted / served_pop if served_pop > 0 else 0
        utilization = st['demand'] / st['cap'] if st['cap'] > 0 else 0

        print(f"{st['name']}({['','小型','中型','大型'][st['scale']]})\t"
            f"{','.join(served_communities) if served_communities else '无'}\t"
            f"{served_pop}\t"
            f"{utilization:.2%}\t"
            f"{avg_sat:.4f}")

    # 整体覆盖率与满意度
    coverage = covered / total_P
    total_sat_all = 0.0
    for k, st in assign.items():
        i = st['idx']
        d = dist_mat_opt[i][k]
        util = st['demand'] / st['cap']
        sat_val = satisfaction(d, util)
        total_sat_all += P[k] * sat_val
    avg_sat_all = total_sat_all / covered if covered > 0 else 0
    print(f"\n整体覆盖率: {coverage:.2%}")
    print(f"整体平均满意度: {avg_sat_all:.4f}")

print("IterableGRPO+Transformer(DSA):")
solution = [0, 3, 2, 0, 1, 3, 0, 0, 0, 0]
exc(solution)
print("枚举:")
solution = [3, 1, 0, 0, 0, 0, 0, 0, 3, 2]
exc(solution)