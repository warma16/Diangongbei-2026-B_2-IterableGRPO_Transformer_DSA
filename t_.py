import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# ================= Floyd 最短距离矩阵 =================
def floyd(graph):
    n = len(graph)
    dist = graph.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

# 假设 exp_dist_mat 已从 two_one.py 导入
from two_one import exp_dist_mat, P, D, total_P
dist_mat_opt = exp_dist_mat
can_serve = (dist_mat_opt <= 1000)

# ================= MDS 生成坐标 =================
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
coords = mds.fit_transform(dist_mat_opt)

# ================= 计算缩放因子 =================
actual_distances = []
mds_distances = []
for i in range(10):
    for j in range(i+1, 10):
        if dist_mat_opt[i][j] > 0:
            actual_distances.append(dist_mat_opt[i][j])
            mds_distances.append(np.linalg.norm(coords[i] - coords[j]))
scale_factor = np.mean(actual_distances) / np.mean(mds_distances)
radius_mds = 1000 / scale_factor

# ================= 两个方案 =================
# 暴力遍历法最优解: [(1, 'A', 3), (2, 'B', 1), (9, 'I', 3), (10, 'J', 2)]
baseline_solution = [3, 1, 0, 0, 0, 0, 0, 0, 3, 2]
baseline_label = "暴力遍历法最优解"
baseline_stations = [0,1,8,9]  # ABIJ

# GRPO+Transformer最优解:  [(2, 3), (3, 2), (5, 1), (6, 3)] 
grpo_solution = [0, 3, 2, 0, 1, 3, 0, 0, 0, 0]
grpo_label = "GRPO+Transformer最优解"
grpo_stations = [1, 2, 4,5]  

# ================= 计算覆盖情况 =================
def get_covered(solution, stations):
    covered = np.zeros(10, dtype=bool)
    for k in range(10):
        for i in stations:
            if can_serve[i][k]:
                covered[k] = True
                break
    return covered

baseline_covered = get_covered(baseline_solution, baseline_stations)
grpo_covered = get_covered(grpo_solution, grpo_stations)

# ================= 绘图 =================
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

# 公共参数
station_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F39C12']

for ax_idx, (solution, stations, covered, label) in enumerate([
    (baseline_solution, baseline_stations, baseline_covered, baseline_label),
    (grpo_solution, grpo_stations, grpo_covered, grpo_label)
]):
    ax = axes[ax_idx]

    # 绘制服务站（大号五角星）
    for idx, i in enumerate(stations):
        ax.scatter(coords[i, 0], coords[i, 1],
                   marker='*', s=600,
                   c=station_colors[idx % len(station_colors)],
                   edgecolors='black', linewidths=1.5,
                   zorder=10, label=f'服务站 {chr(65+i)}')

    # 绘制被覆盖小区（绿色圆点）和未覆盖小区（红色叉号）
    for k in range(10):
        if covered[k]:
            ax.scatter(coords[k, 0], coords[k, 1],
                       marker='o', s=350,
                       c='#2ECC71', edgecolors='black', linewidths=1.2,
                       zorder=9, alpha=0.85)
        else:
            ax.scatter(coords[k, 0], coords[k, 1],
                       marker='X', s=400,
                       c='#E74C3C', edgecolors='black', linewidths=1.2,
                       zorder=8, alpha=0.7)

    # 服务覆盖连线
    connection_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F39C12']
    for idx, i in enumerate(stations):
        for k in range(10):
            if can_serve[i][k]:
                ax.plot([coords[i, 0], coords[k, 0]],
                        [coords[i, 1], coords[k, 1]],
                        linestyle='--', linewidth=1.5,
                        color=connection_colors[idx % len(connection_colors)],
                        alpha=0.35, zorder=1)

    # 服务半径圈
    for idx, i in enumerate(stations):
        circle = plt.Circle((coords[i, 0], coords[i, 1]), radius_mds,
                            fill=False, color=connection_colors[idx % len(connection_colors)],
                            linestyle='-', linewidth=2, alpha=0.45)
        ax.add_patch(circle)

    # 小区标签
    for k in range(10):
        label = chr(65 + k)
        ax.annotate(label, (coords[k, 0], coords[k, 1]),
                    xytext=(coords[k, 0] + 0.015, coords[k, 1] + 0.015),
                    fontsize=13, fontweight='bold',
                    ha='center', va='center')

    # 统计信息
    covered_count = sum(covered)
    covered_pop = sum(P[k] for k in range(10) if covered[k])
    coverage_rate = covered_pop / total_P * 100

    scale_names = {0: '不建', 1: '小型', 2: '中型', 3: '大型'}
    station_desc = ' + '.join([f"{chr(65+i)}({scale_names[solution[i]]})" for i in stations])

    ax.set_title(f'{label}\n方案：{station_desc} | '
                 f'覆盖小区：{covered_count}/10 | 地理覆盖率：{coverage_rate:.1f}%',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('MDS 维度 1', fontsize=12)
    ax.set_ylabel('MDS 维度 2', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

# 总标题
fig.suptitle('养老服务站最优方案覆盖情况对比', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('coverage_comparison.png', dpi=200, bbox_inches='tight')
print("覆盖情况对比图已保存为 coverage_comparison.png")
plt.show()