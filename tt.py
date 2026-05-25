def get_S3_by_service_type(service_type: str, price: float) -> float:
    """根据服务类型和实际价格，返回 S3 评分"""
    match service_type:
        case "助餐":
            baseline_price = 8
            price=10
        case "日间照料":
            baseline_price = 16
            price=20
        case "上门护理":
            baseline_price = 24
            price=30
        case "康复理疗":
            baseline_price = 23
            price=28
        case "助浴":
            baseline_price = 20
            price=25
        case "紧急救助":
            baseline_price = 8   
            price=0
            # 营收0，但基准价8，实际price=0时≤基准，S3=1.00
        case _:
            raise ValueError(f"未知服务类型: {service_type}")

    price_ratio = price / baseline_price
    # 根据价格区间确定 S3
    if price <= baseline_price:
        S3 = 1.00
    elif price_ratio <= 1.10:
        S3 = 0.90
    elif price_ratio <= 1.20:
        S3 = 0.75
    else:
        S3 = 0.60
    return S3