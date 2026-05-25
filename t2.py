import pandas as pd
import numpy as np
from two_one import P, D, building_costs_, capacities, exp_dist_mat, exp_1_3_actual
from floyd import floyd

# ================= 满意度函数 =================
WEIGHTS = (0.2, 0.3, 0.5)

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

    s3 = 1.00
    return WEIGHTS[0] * s1 + WEIGHTS[1] * s2 + WEIGHTS[2] * s3

# ================= 加载服务价格与成本数据 =================
# ================= 加载服务价格与成本数据 =================
# 用 header=1 跳过第一行总标题，让第二行成为列名
service_price_df = pd.read_excel("./table2.xlsx", sheet_name="服务营收及支出", header=1)
operation_cost_df = pd.read_excel("./table3.xlsx", sheet_name="服务站建设与运营成本", header=1)

# 确认列名（可选）
#print(service_price_df.columns.tolist())   # 应显示 ['服务项目', '单次服务营收（元）', '单次服务直接支出（元）（基准价格）']
#print(operation_cost_df.columns.tolist())  # 应显示 ['站点规模', '一次性建设成本（万元）', '日均固定管理成本（元/日）', '日最大服务人次']

service_names = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']

# 服务单价（元/次）和单位成本（元/次）
prices = {}
unit_costs = {}
for sv in service_names:
    row = service_price_df.loc[service_price_df['服务项目'] == sv]
    #print(row)
    #exit()
    prices[sv] = row['单次服务营收（元） '].values[0]    # 注意加个空格
    unit_costs[sv] = row['单次服务直接支出（元）（基准价格）'].values[0]  # 这个没空格

# 日固定管理成本（元/日）
daily_op_cost = {
    1: operation_cost_df.loc[operation_cost_df['站点规模'] == '小型', '日均固定管理成本（元/日）'].values[0],
    2: operation_cost_df.loc[operation_cost_df['站点规模'] == '中型', '日均固定管理成本（元/日）'].values[0],
    3: operation_cost_df.loc[operation_cost_df['站点规模'] == '大型', '日均固定管理成本（元/日）'].values[0],
}

# 然后继续后续代码（去掉 exit()）
# ================= 最优方案 =================
def exc(solution):
    scale_names = {1: '小型', 2: '中型', 3: '大型'}

    # ================= 准备距离和可达性 =================
    dist_mat_opt = floyd(exp_dist_mat)
    can_serve = (dist_mat_opt <= 1000)

    # ================= 构建各小区各项服务的月均需求次数 =================
    # 从问题1.3结果中提取“_合计”列
    monthly_demand_matrix = np.zeros((10, len(service_names)))
    for idx, sv in enumerate(service_names):
        col = f'{sv}_合计'
        monthly_demand_matrix[:, idx] = exp_1_3_actual[col].values

    # ================= 初始化站点信息 =================
    station_info = []
    for i in range(10):
        s = solution[i]
        if s > 0:
            station_info.append({
                'idx': i,
                'name': chr(65 + i),
                'scale': s,
                'cap': capacities[s - 1],
                'remain': capacities[s - 1],
                'demand': 0.0,
                'monthly_demand': {sv: 0.0 for sv in service_names}
            })

    # ================= 贪心分配 =================
    assign = {}  # k -> station
    for k in range(10):
        if P[k] == 0:
            continue
        candidates = []
        for st in station_info:
            i = st['idx']
            if can_serve[i][k]:
                sat = satisfaction(dist_mat_opt[i][k], 0)
                candidates.append((sat, st))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, st in candidates:
            if st['remain'] >= D[k]:
                st['remain'] -= D[k]
                st['demand'] += D[k]
                assign[k] = st
                # 累加月需求次数
                for sv_idx, sv in enumerate(service_names):
                    st['monthly_demand'][sv] += monthly_demand_matrix[k, sv_idx]
                break

    # ================= 计算各站点年度利润 =================
    results = []
    for st in station_info:
        # 年营收（万元）
        annual_revenue = 0.0
        #print("opopopo")
        for sv in service_names:
            monthly_times = st['monthly_demand'][sv]
            p_tmp=prices[sv]
            #print(p_tmp)
            if "（公益免费）" in str(p_tmp):
                p_tmp=0
            annual_revenue += monthly_times * p_tmp * 12 / 10000

        # 年服务变动成本（万元）
        annual_service_cost = 0.0
        for sv in service_names:
            monthly_times = st['monthly_demand'][sv]
            annual_service_cost += monthly_times * unit_costs[sv] * 12 / 10000

        # 年运营成本（万元）
        annual_op_cost = daily_op_cost[st['scale']] * 365 / 10000

        annual_total_cost = annual_op_cost + annual_service_cost
        annual_profit = annual_revenue - annual_total_cost

        results.append({
            '站点': st['name'],
            '规模': scale_names[st['scale']],
            '年度营收（万元）': round(annual_revenue, 2),
            '年度总成本（万元）': round(annual_total_cost, 2),
            '年度利润（万元）': round(annual_profit, 2)
        })

    # 输出表格
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))


print("IterableGRPO+Transformer(DSA):")
solution = [0, 3, 2, 0, 1, 3, 0, 0, 0, 0]
exc(solution)
print("枚举:")
solution = [3, 1, 0, 0, 0, 0, 0, 0, 3, 2]
exc(solution)