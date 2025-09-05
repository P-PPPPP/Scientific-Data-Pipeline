import os
from utils.data_processor import data_completeness_processor


raw_data_directory = "/media/hdd/ppeng/storage/sz_weather/daily_data_raw/"
save_directory = '/media/hdd/ppeng/storage/sz_weather/filled_data/'
grid_info_path = "/media/hdd/ppeng/storage/sz_weather/grid_info.csv"
stats_info_path = "/media/hdd/ppeng/storage/sz_weather/daily_data_raw/0_completeness_stats.csv"

# 其余变量的插值
interpolate_configs = {
    'scaler': {
        'columns': ['T', 'MAXTOFDAY', 'SLP', 'RHSFC', 'V', 'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H'],
        'methods': {
            'T': {'method': 'cubic'},                   # 温度（摄氏度）
            'MAXTOFDAY': {'method': 'cubic'},           # 日最高温度（摄氏度）
            'SLP': {'method': 'cubic'},                 # 气压（百帕）
            'RHSFC': {'method': 'spline', 'order': 2},  # 相对湿度（百分比）
            'V': {'method': 'linear'},                  # 能见度（公里）
            'RAIN01H': {'method': 'linear'},
            'RAIN02H': {'method': 'linear'},
            'RAIN03H': {'method': 'linear'},
            'RAIN06H': {'method': 'linear'},
            'RAIN24H': {'method': 'linear'}             # 1小时，2小时，3小时，6小时，24小时累计降雨量
        }
    },
    'vector': {
        'columns': ['WSPD', 'WDIR', 'WD3SMAXDF', 'WD3SMAXDD'],
        'relationship': [{
            'type': 'scaler-direction',
            'direct-sys': 'angle',
            'scaler': 'WSPD',           # 风速（米/秒）
            'vector': 'WDIR',           # 风向（度）
            'method': 'linear'
            }, {
            'type': 'scaler-direction',
            'direct-sys': 'angle',
            'scaler': 'WD3SMAXDF',      # 极大风速（米/秒）
            'vector': 'WD3SMAXDD',      # 极大风向（度）
            'method': 'linear'
        }]
    }
}

# 实例化
data_completting_processor = data_completeness_processor(
    raw_data_directory=raw_data_directory,
    save_directory=save_directory,
    grid_info_path=grid_info_path,
    stats_info_path=stats_info_path,
    interpolate_configs=interpolate_configs,
    max_grid_id=4231
)

''' 处理全部数据 '''
data_completting_processor.process_all_data_concurrently(n_processes=8)
raise KeyboardInterrupt

''' 处理单个数据 '''
date_str = '20200101'
# date_str = '20250901'
# 读取
df, _, _ = data_completting_processor.process_single_date(date_str)

print(df)