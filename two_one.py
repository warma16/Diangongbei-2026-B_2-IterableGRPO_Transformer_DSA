import numpy as np
import pandas as pd
import sympy as sp
from prefab import exp_1_3_actual


# ============================================================
# 第一部分：数据加载（与你之前相同）
# ============================================================
print("==== 基础数据加载 ====")
print(exp_1_3_actual[:20])
total_columns = exp_1_3_actual.filter(like='_合计')
print(total_columns)

site_infos_ = pd.read_excel("./table3.xlsx")
print(site_infos_[:4])

site_infos = {}
site_type = ["small","medium","large"]
site_cost_type = ["building_cost","running_cost_per_day","capacity"]
for i in range(3):
    t = site_type[i]
    i_ = i+1
    for j in range(3):
        j_ = j+1
        sct = site_cost_type[j]
        if t not in site_infos:
            site_infos[t] = {}
        site_infos[t][sct] = site_infos_.iloc[i_, j_]

building_costs_ = [
    site_infos["small"]["building_cost"],
    site_infos["medium"]["building_cost"],
    site_infos["large"]["building_cost"]
]
capacities = [
    site_infos["small"]["capacity"],
    site_infos["medium"]["capacity"],
    site_infos["large"]["capacity"]
]

# 距离矩阵
location_infos_ = pd.read_excel("./table4.xlsx")
dist_mat = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        dist_mat[i][j] = location_infos_.iloc[j+1, i+1]

exp_dist_mat=dist_mat

can_be_select_M = (dist_mat <= 1000).tolist()

P = (exp_1_3_actual['自理人数'] + exp_1_3_actual['半失能人数'] + exp_1_3_actual['失能人数']).values

# 提取所有服务类别的合计列，求和得到月均总需求次数（次/月）
total_columns = exp_1_3_actual.filter(like='_合计')
monthly_demand = total_columns.sum(axis=1).values

# 将月均需求转换为日均需求（次/日）
D = monthly_demand / 30

# 打印验证
print("各小区老人总数（人）:", P)
print("各小区日均总需求（人次/日）:", D)

total_P = sum(P)

# ============================================================
# 第二部分：决策变量（修正 Z 的定义方式）
# ============================================================
print("\n==== 定义决策变量 ====")

X = sp.Matrix([
    [sp.symbols(f"small_{i}", integer=True),
     sp.symbols(f"medium_{i}", integer=True),
     sp.symbols(f"large_{i}", integer=True)]
    for i in range(1, 11)
])  # 10x3

# 修正点：用 sp.zeros 创建 10x10 矩阵，再逐个填充符号
Z = sp.Matrix([
    [sp.symbols(f"Z_{i+1}_{k+1}", integer=True) for k in range(10)]
    for i in range(10)
])  # 10x10

# ============================================================
# 第三部分：约束条件
# ============================================================
print("\n==== 构建约束条件 ====")

constraints = []

# 1. 每个小区最多建一个站
for i in range(10):
    constraints.append(sum(X[i, :]) <= 1)

# 2. 变量范围 0/1（分开写避免 sp.And 的问题）
for i in range(10):
    for j in range(3):
        print(X[i,j])
        constraints.append(X[i, j] >= 0)
        constraints.append(X[i, j] <= 1)
    for k in range(10):
        print(Z[i,k])
        constraints.append(Z[i, k] >= 0)
        constraints.append(Z[i, k] <= 1)

# 3. 预算约束
building_costs_vec = sp.Matrix(building_costs_)
total_cost_expr = sum(X * building_costs_vec)
constraints.append(total_cost_expr <= 120)

# 4. 服务半径约束：距离>1000 则 Z[i][k]=0
for i in range(10):
    for k in range(10):
        if dist_mat[i][k] > 1000:
            constraints.append(sp.Eq(Z[i, k], 0))

# 5. 每个小区 k 的老人必须被分配给一个可达的服务站
for k in range(10):
    valid_i = [i for i in range(10) if can_be_select_M[i][k]]
    if valid_i:
        constraints.append(sum(Z[i, k] for i in valid_i) == 1)
    else:
        for i in range(10):
            constraints.append(sp.Eq(Z[i, k], 0))

# 6. 只有建站的小区才能接收分配
for i in range(10):
    has_station = sum(X[i, :])
    for k in range(10):
        constraints.append(Z[i, k] <= has_station)

# 7. 容量约束
for i in range(10):
    total_demand_to_i = sum(Z[i, k] * D[k] for k in range(10))
    cap_i = sp.Piecewise(
        (0, sp.Eq(sum(X[i, :]), 0)),
        (capacities[0], sp.Eq(X[i, 0], 1)),
        (capacities[1], sp.Eq(X[i, 1], 1)),
        (capacities[2], sp.Eq(X[i, 2], 1))
    )
    constraints.append(total_demand_to_i <= cap_i)

# ============================================================
# 第四部分：目标函数
# ============================================================
if __name__ == "main":
    print("\n==== 构建目标函数 ====")

    allocated_P = sum(Z[i, k] * P[k] for i in range(10) for k in range(10))
    coverage = allocated_P / total_P

    # 符号化满意度函数
    def symbolic_satisfaction(dist_km, utilization_rate):
        d = dist_km
        S1 = sp.Piecewise(
            (1.00, d <= 300),
            (0.90, d <= 500),
            (0.75, d <= 650),
            (0.60, True)
        )
        r = utilization_rate
        S2 = sp.Piecewise(
            (1.00, r <= 0.60),
            (0.93, r <= 0.75),
            (0.85, r <= 0.85),
            (0.72, r <= 0.95),
            (0.60, True)
        )
        S3 = 1.0   # 价格暂定基准
        return 0.5*S1 + 0.3*S2 + 0.2*S3

    total_satisfaction = 0
    for i in range(10):
        total_demand_i = sum(Z[i, k] * D[k] for k in range(10))
        cap_i = sp.Piecewise(
            (0, sp.Eq(sum(X[i, :]), 0)),
            (capacities[0], sp.Eq(X[i, 0], 1)),
            (capacities[1], sp.Eq(X[i, 1], 1)),
            (capacities[2], sp.Eq(X[i, 2], 1)),
            (0, True)  # ← 新增默认分支，防止多元条件无法自动补全
        )
        utilization_i = sp.Piecewise(
            (0, sp.Eq(cap_i, 0)),
            (total_demand_i / cap_i, True)
        )
        for k in range(10):
            dist = dist_mat[i][k]
            sat = symbolic_satisfaction(dist, utilization_i)
            total_satisfaction += Z[i, k] * P[k] * sat

    avg_satisfaction = sp.Piecewise(
        (0, sp.Eq(allocated_P, 0)),
        (total_satisfaction / allocated_P, True)
    )

    alpha, beta = 0.5, 0.5
    objective = alpha * coverage + beta * avg_satisfaction

    print("\n目标函数表达式已生成。")

    # ============================================================
    # 第五部分：输出模型摘要
    # ============================================================
    print("\n" + "="*60)
    print("2.1 符号优化模型")
    print("="*60)
    print(f"决策变量：X (10x3) 建站规模, Z (10x10) 分配关系")
    print(f"约束数量：{len(constraints)}")
    print("模型构建完成，可用于后续求解。")
    with open("c.txt","w",encoding="utf-8") as f:
        for c in constraints:
            f.write(str(c))
            f.write("\n")