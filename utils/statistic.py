import pandas as pd
from datetime import datetime, timedelta


def generate_all_time_points(date_str):
    """ 生成指定日期的所有时间点, 每10分钟"""
    # 将日期字符串转换为datetime对象
    base_date = datetime.strptime(date_str, "%Y%m%d")
    
    # 生成全天所有10分钟间隔的时间点
    for i in range(0, 24*6):  # 24小时 * 6个10分钟间隔
        time_point = base_date + timedelta(minutes=10*i)
        yield time_point.strftime("%H:%M")


def analyze_data_completeness(df, target_date_str, max_grid_id):
    """ 分析指定日期数据帧的完整性 """
    # 将 DDATETIME 转换为标准时间格式（假设原始格式是时间戳或字符串）
    df['TIME_POINT'] = pd.to_datetime(df['DDATETIME']).dt.strftime('%H:%M')
    
    # 生成当天所有可能的时间点（每10分钟）
    all_time_points = list(generate_all_time_points(target_date_str))
    
    # 获取数据中实际存在的时间点
    actual_time_points = df['TIME_POINT'].unique()
    
    # 找出缺失的时间点
    missing_time_points = set(all_time_points) - set(actual_time_points)
    
    # 对于每个时间点，检查是否有所有网格点的数据
    incomplete_time_points = {}
    for time_point in actual_time_points:
        time_data = df[df['TIME_POINT'] == time_point]
        present_gridids = set(time_data['GRIDID'].unique())
        expected_gridids = set(range(max_grid_id + 1))
        missing_gridids = expected_gridids - present_gridids
        
        if missing_gridids:
            incomplete_time_points[time_point] = len(missing_gridids)
    
    # 计算统计数据
    total_time_points = len(all_time_points)
    complete_time_points = total_time_points - len(missing_time_points) - len(incomplete_time_points)
    expected_points = total_time_points * (max_grid_id + 1)
    actual_points = len(df)
    completeness_ratio = actual_points / expected_points
    
    # 创建完整性统计信息
    completeness_info = {
        'date': target_date_str,
        'total_time_points': total_time_points,
        'complete_time_points': complete_time_points,
        'incomplete_time_points_count': len(incomplete_time_points),
        'missing_time_points_count': len(missing_time_points),
        'expected_points': expected_points,
        'actual_points': actual_points,
        'completeness_ratio': completeness_ratio,
        'missing_time_points': sorted(missing_time_points),
        'incomplete_time_points': incomplete_time_points
    }
    
    # 将统计信息保存到类属性中
    return completeness_info