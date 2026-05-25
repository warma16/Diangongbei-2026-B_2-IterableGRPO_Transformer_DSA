import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 全局设置中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# ==================== 全局参数设置 ====================
死亡率 = 0.05      # 年均自然死亡率 5%
新增率 = 0.07      # 年均新增老年人率 7%
新增率 = 0.08 #（4.1）
p_ZB = 0.045       # 自理→半失能转移概率 4.5% (从附件1获取)
p_ZB = 0.055#（4.1）
p_BS = 0.10        # 半失能→失能转移概率 10% (从附件1获取)
p_BS = 0.095#（4.1）

# ==================== 1. 数据准备 ====================

# 附件1：人口与老人结构数据
小区数据 = {
    '小区': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    '总人口': [3200, 2800, 4100, 2500, 3600, 2200, 3900, 2600, 3400, 3000],
    '60+老人数': [712, 608, 920, 544, 784, 472, 864, 568, 736, 656],
    '自理老人': [496, 408, 632, 368, 536, 328, 592, 392, 504, 456],
    '半失能老人': [152, 136, 208, 120, 176, 104, 192, 128, 168, 144],
    '失能老人': [64, 64, 80, 56, 72, 40, 80, 48, 64, 56],
    '人均月收入': [3400, 3100, 3800, 2900, 3500, 2700, 3600, 3000, 3300, 3200]
}

df_initial = pd.DataFrame(小区数据)
print("========== 初始数据 ==========")
print(df_initial.to_string(index=False))

# 附件2：每位老人月均服务需求次数（次/月）
服务需求 = {
    '服务项目': ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助'],
    '自理': [14, 8, 0, 2, 0, 0.15],
    '半失能': [20, 14, 6, 4, 2, 1],
    '失能': [22, 18, 12, 6, 4, 3]
}
df_service_demand = pd.DataFrame(服务需求)
print("\n========== 服务需求数据 ==========")
print(df_service_demand.to_string(index=False))

# 服务单价（需要从附件2中获取，这里假设数据）
# 注意：实际附件2中应该有营收数据，这里使用示例单价
服务单价 = {
    '助餐': 8,
    '日间照料': 15,
    '上门护理': 25,
    '康复理疗': 20,
    '助浴': 30,
    '紧急救助': 50
}

# 消费上限比例
消费上限比例 = {
    '自理': 0.20,      # ≤ 20%
    '半失能': 0.25,    # ≤ 25%
    '失能': 0.30       # ≤ 30%
}

# ==================== 2. 问题1.1：预测模型 ====================

def predict_population(df, years=5):
    """
    预测未来years年各小区各类老人数量
    
    参数:
        df: DataFrame，包含初始老人数据
        years: 预测年数
    
    返回:
        results: 字典，包含每年每个小区的数据
    """
    
    # 提取初始数据
    小区列表 = df['小区'].tolist()
    初始自理 = df['自理老人'].values.astype(float)
    初始半失能 = df['半失能老人'].values.astype(float)
    初始失能 = df['失能老人'].values.astype(float)
    
    # 存储每年每个小区的数据
    # 格式: results[年份][小区索引] = (自理, 半失能, 失能)
    results = {0: list(zip(初始自理, 初始半失能, 初始失能))}
    
    for year in range(1, years + 1):
        当年数据 = []
        prev_data = results[year - 1]
        
        for i in range(len(小区列表)):
            Z_prev, B_prev, S_prev = prev_data[i]
            N_prev = Z_prev + B_prev + S_prev  # 前一年总老年人口
            
            # 步骤1：自然死亡（所有老人减少5%）+ 新增老人（占当前总人口的7%）
            Z_after_death_new = Z_prev * (1 - 死亡率) + 新增率 * N_prev
            B_after_death = B_prev * (1 - 死亡率)
            S_after_death = S_prev * (1 - 死亡率)
            
            # 步骤2：状态转移
            # 自理→半失能
            Z_to_B = Z_after_death_new * p_ZB
            Z_new = Z_after_death_new - Z_to_B
            
            # 半失能→失能
            B_to_S = B_after_death * p_BS
            B_new = B_after_death - B_to_S + Z_to_B
            S_new = S_after_death + B_to_S
            
            # 确保没有负值
            Z_new = max(0, Z_new)
            B_new = max(0, B_new)
            S_new = max(0, S_new)
            
            当年数据.append((Z_new, B_new, S_new))
        
        results[year] = 当年数据
    
    return results, 小区列表

# 执行预测
pred_results, 小区列表 = predict_population(df_initial, years=5)

# 输出问题1.1结果
print("\n" + "="*80)
print("问题1.1：未来五年各小区各类老人数量预测结果")
print("="*80)

for year in range(1, 6):
    print(f"\n--- 第{year}年末 ---")
    print(f"{'小区':<6} {'自理老人':<12} {'半失能老人':<12} {'失能老人':<12} {'总人数':<12}")
    print("-"*54)
    for i, 小区 in enumerate(小区列表):
        Z, B, S = pred_results[year][i]
        print(f"{小区:<6} {round(Z):<12} {round(B):<12} {round(S):<12} {round(Z+B+S):<12}")

# ==================== 3. 问题1.2：理论月需求预测 ====================

def calculate_theoretical_demand(pred_results, 小区列表, df_service_demand, year=5):
    """
    计算第year年末各小区各项服务的理论月需求次数
    
    参数:
        pred_results: 预测结果
        小区列表: 小区名称列表
        df_service_demand: 服务需求数据
        year: 目标年份
    
    返回:
        df_demand: DataFrame，包含各项服务需求
    """
    
    # 准备数据结构
    服务项目列表 = df_service_demand['服务项目'].tolist()
    数据行 = []
    
    for i, 小区 in enumerate(小区列表):
        Z, B, S = pred_results[year][i]
        行数据 = {'小区': 小区, '自理人数': round(Z), '半失能人数': round(B), '失能人数': round(S)}
        
        for _, row in df_service_demand.iterrows():
            服务 = row['服务项目']
            # 计算该项服务的总需求
            z_demand = round(Z * row['自理'])
            b_demand = round(B * row['半失能'])
            s_demand = round(S * row['失能'])
            行数据[f'{服务}_合计'] = z_demand + b_demand + s_demand
            行数据[f'{服务}_自理'] = z_demand
            行数据[f'{服务}_半失能'] = b_demand
            行数据[f'{服务}_失能'] = s_demand
        
        数据行.append(行数据)
    
    df_demand = pd.DataFrame(数据行)
    return df_demand

# 计算理论需求
df_theoretical = calculate_theoretical_demand(pred_results, 小区列表, df_service_demand, year=5)

print("\n" + "="*80)
print("问题1.2：第5年末各小区各项服务理论月需求次数（未考虑消费约束）")
print("="*80)

# 输出汇总表
汇总列 = ['小区'] + [f'{服务}_合计' for 服务 in df_service_demand['服务项目']]
print(df_theoretical[汇总列].to_string(index=False))

# ==================== 4. 问题1.3：考虑消费约束后的实际需求 ====================

def calculate_actual_demand(pred_results, 小区列表, df_initial, df_service_demand, 
                           服务单价, 消费上限比例, year=5):
    """
    计算第year年末考虑消费约束后的实际月需求
    """
    
    服务项目列表 = df_service_demand['服务项目'].tolist()
    数据行 = []
    
    for i, 小区 in enumerate(小区列表):
        Z, B, S = pred_results[year][i]
        人均收入 = df_initial.loc[df_initial['小区'] == 小区, '人均月收入'].values[0]
        
        Z = round(Z)
        B = round(B)
        S = round(S)
        
        行数据 = {'小区': 小区, '自理人数': Z, '半失能人数': B, '失能人数': S}
        
        # 计算各类老人的理论月费用
        for 类型, 人数, 类型名 in [('自理', Z, '自理'), ('半失能', B, '半失能'), ('失能', S, '失能')]:
            if 人数 == 0:
                continue
            
            # 计算该类型老人的理论月费用
            理论费用 = 0
            for _, row in df_service_demand.iterrows():
                服务 = row['服务项目']
                单价 = 服务单价[服务]
                需求次数 = row[类型]
                理论费用 += 人数 * 需求次数 * 单价
            
            人均费用 = 理论费用 / 人数
            
            # 计算消费上限
            上限 = 人均收入 * 消费上限比例[类型]
            
            # 判断是否需要削减
            if 人均费用 > 上限:
                削减比例 = 上限 / 人均费用
            else:
                削减比例 = 1.0
            
            # 保存该类型老人的削减比例
            行数据[f'{类型名}_削减比例'] = round(削减比例, 4)
            行数据[f'{类型名}_人均理论费用'] = round(人均费用, 2)
            行数据[f'{类型名}_人均上限'] = round(上限, 2)
        
        # 计算各项服务的实际需求
        for _, row in df_service_demand.iterrows():
            服务 = row['服务项目']
            
            # 各类老人的理论需求
            z_demand = Z * row['自理']
            b_demand = B * row['半失能']
            s_demand = S * row['失能']
            
            # 应用削减
            z_实际 = z_demand * 行数据.get('自理_削减比例', 1.0)
            b_实际 = b_demand * 行数据.get('半失能_削减比例', 1.0)
            s_实际 = s_demand * 行数据.get('失能_削减比例', 1.0)
            
            行数据[f'{服务}_自理'] = round(z_实际)
            行数据[f'{服务}_半失能'] = round(b_实际)
            行数据[f'{服务}_失能'] = round(s_实际)
            行数据[f'{服务}_合计'] = round(z_实际 + b_实际 + s_实际)
        
        数据行.append(行数据)
    
    df_actual = pd.DataFrame(数据行)
    return df_actual

# 计算实际需求
df_actual = calculate_actual_demand(pred_results, 小区列表, df_initial, 
                                     df_service_demand, 服务单价, 消费上限比例, year=5)

exp_1_3_actual=df_actual

print("\n" + "="*80)
print("问题1.3：第5年末各小区各项服务实际月需求次数（考虑消费约束）")
print("="*80)

# 输出削减信息
print("\n--- 消费约束检查 ---")
print(f"{'小区':<6} {'自理上限':<12} {'自理费用':<12} {'需削减':<8} {'半失上限':<12} {'半失费用':<12} {'需削减':<8} {'失能上限':<12} {'失能费用':<12} {'需削减':<8}")
print("-"*100)
for _, row in df_actual.iterrows():
    print(f"{row['小区']:<6} {row.get('自理_人均上限', 0):<12.2f} {row.get('自理_人均理论费用', 0):<12.2f} "
          f"{'是' if row.get('自理_削减比例', 1) < 1 else '否':<8} "
          f"{row.get('半失能_人均上限', 0):<12.2f} {row.get('半失能_人均理论费用', 0):<12.2f} "
          f"{'是' if row.get('半失能_削减比例', 1) < 1 else '否':<8} "
          f"{row.get('失能_人均上限', 0):<12.2f} {row.get('失能_人均理论费用', 0):<12.2f} "
          f"{'是' if row.get('失能_削减比例', 1) < 1 else '否':<8}")

# 输出实际需求汇总
print("\n--- 各项服务实际月需求汇总 ---")
汇总列 = ['小区'] + [f'{服务}_合计' for 服务 in df_service_demand['服务项目']]
print(df_actual[汇总列].to_string(index=False))

# ==================== 5. 详细输出（分老人类型） ====================

print("\n" + "="*80)
print("详细输出：各服务项目按老人类型分（理论 vs 实际）")
print("="*80)

for 服务 in df_service_demand['服务项目']:
    print(f"\n--- {服务} ---")
    print(f"{'小区':<6} {'理论自理':<10} {'理论半失':<10} {'理论失能':<10} {'理论合计':<10} "
          f"{'实际自理':<10} {'实际半失':<10} {'实际失能':<10} {'实际合计':<10}")
    print("-"*80)
    for i, 小区 in enumerate(小区列表):
        理论值 = df_theoretical.loc[i]
        实际值 = df_actual.loc[i]
        print(f"{小区:<6} {理论值[f'{服务}_自理']:<10} {理论值[f'{服务}_半失能']:<10} "
              f"{理论值[f'{服务}_失能']:<10} {理论值[f'{服务}_合计']:<10} "
              f"{实际值[f'{服务}_自理']:<10} {实际值[f'{服务}_半失能']:<10} "
              f"{实际值[f'{服务}_失能']:<10} {实际值[f'{服务}_合计']:<10}")

# ==================== 6. 可视化 ====================

def plot_population_trends(pred_results, 小区列表):
    """绘制各小区人口变化趋势"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, 小区 in enumerate(小区列表):
        ax = axes[i]
        years = [f'Y{j}' for j in range(6)]  # 包括第0年
        
        z_values = [round(pred_results[y][i][0]) for y in range(6)]
        b_values = [round(pred_results[y][i][1]) for y in range(6)]
        s_values = [round(pred_results[y][i][2]) for y in range(6)]
        total_values = [z_values[j] + b_values[j] + s_values[j] for j in range(6)]
        
        ax.plot(years, z_values, 'g-o', label='自理', linewidth=2)
        ax.plot(years, b_values, 'b-s', label='半失能', linewidth=2)
        ax.plot(years, s_values, 'r-^', label='失能', linewidth=2)
        ax.plot(years, total_values, 'k--', label='总人数', linewidth=2, alpha=0.6)
        
        ax.set_title(f'小区 {小区}', fontsize=12, fontweight='bold')
        ax.set_xlabel('年份')
        ax.set_ylabel('人数')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('各小区未来五年老人数量变化趋势', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 运行可视化
#plot_population_trends(pred_results, 小区列表)

# ==================== 7. 服务需求对比图 ====================

def plot_demand_comparison(df_theoretical, df_actual, 小区列表):
    """绘制理论需求和实际需求对比"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, 小区 in enumerate(小区列表):
        ax = axes[i]
        服务项目 = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
        
        理论值 = [df_theoretical.loc[i, f'{s}_合计'] for s in 服务项目]
        实际值 = [df_actual.loc[i, f'{s}_合计'] for s in 服务项目]
        
        x = np.arange(len(服务项目))
        width = 0.35
        
        ax.bar(x - width/2, 理论值, width, label='理论需求', color='skyblue', alpha=0.8)
        ax.bar(x + width/2, 实际值, width, label='实际需求', color='salmon', alpha=0.8)
        
        ax.set_title(f'小区 {小区}', fontsize=12, fontweight='bold')
        ax.set_xlabel('服务项目')
        ax.set_ylabel('月需求次数')
        ax.set_xticks(x)
        ax.set_xticklabels(服务项目, rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('第5年末各小区服务需求对比（理论 vs 实际）', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 运行对比图
#plot_demand_comparison(df_theoretical, df_actual, 小区列表)

print("\n========== 计算完成 ==========")




