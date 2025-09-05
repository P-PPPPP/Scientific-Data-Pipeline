import os
import re
import glob
import swifter
import concurrent.futures
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from scipy.interpolate import Rbf
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from utils.statistic import analyze_data_completeness
from utils.functions import safe_lookup


class daily_data_processor:
    def __init__(self, data_dir, max_workers=4, save_dir='./daily_data', max_grid_id=4231):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.max_workers = max_workers
        self.max_grid_id = max_grid_id
        self.all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')] # 所有 csv 文件列表
        self.completeness_stats = defaultdict(dict)

    @staticmethod
    def extract_time_from_filename(filename):
        ''' 从文件名中提取开始时间和结束时间. 文件名格式: page%d_rows10000_%y%m%d_%H%M%Sto%y%m%d_%H%M%S.csv '''
        pattern = r'page\d+_rows10000_(\d+)_(\d+)to(\d+)_(\d+)\.csv'
        match = re.search(pattern, filename)
        if match:
            start_date_str = match.group(1)
            start_time_str = match.group(2)
            end_date_str = match.group(3)
            end_time_str = match.group(4)
            start_datetime = datetime.strptime(f"{start_date_str}_{start_time_str}", "%Y%m%d_%H%M%S")
            end_datetime = datetime.strptime(f"{end_date_str}_{end_time_str}", "%Y%m%d_%H%M%S")
            return start_datetime, end_datetime
        return None, None
    
    def get_all_date_range(self):
        ''' 获取所有CSV文件中的最小和最大时间 '''
        all_files = glob.glob(os.path.join(self.data_dir, "page*_rows10000_*.csv"))
        min_time = None
        max_time = None
        for file in all_files:
            start_time, end_time = self.extract_time_from_filename(os.path.basename(file))
            if start_time and end_time:
                if min_time is None or start_time < min_time:
                    min_time = start_time
                if max_time is None or end_time > max_time:
                    max_time = end_time
        return min_time, max_time

    def save_completeness_stats_to_csv(self, file_path):
        """ 将完整性统计数据保存为CSV文件 """
        try:
            # 创建摘要数据
            summary_data = []
            date_str_list = sorted(self.completeness_stats.keys())
            for date_str in date_str_list:
                stats = self.completeness_stats[date_str]
                row = {
                    'date': date_str,
                    'total_time_points': stats['total_time_points'],
                    'complete_time_points': stats['complete_time_points'],
                    'incomplete_time_points_count': stats['incomplete_time_points_count'],
                    'missing_time_points_count': stats['missing_time_points_count'],
                    'expected_points': stats['expected_points'],
                    'actual_points': stats['actual_points'],
                    'completeness_ratio': stats['completeness_ratio']
                }
                summary_data.append(row)
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(summary_data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            print(f"完整性统计数据已保存到: {file_path}")
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")

    def match_files_for_date(self, target_date_str):
        ''' 匹配指定日期的所有文件，返回文件列表及其时间范围 '''
        # 用于存储匹配到的文件及其时间范围
        matched_files = []
        
        # 目标日期对应的datatime格式开始和结束时间
        target_date = datetime.strptime(target_date_str, '%Y%m%d')   
        start_of_day = target_date
        end_of_day = target_date + timedelta(days=1) - timedelta(seconds=1)
        
        # 正则表达式匹配文件名中的时间范围（考虑"page%d_rows10000_"前缀）
        pattern = r'page\d+_rows10000_(\d{8}_\d{6})to(\d{8}_\d{6})\.csv'
        
        for file in self.all_files:
            match = re.match(pattern, file)
            if match:
                start_str, end_str = match.groups()
                file_start = datetime.strptime(start_str, '%Y%m%d_%H%M%S')
                file_end = datetime.strptime(end_str, '%Y%m%d_%H%M%S')
                
                # 检查文件时间范围是否与目标日期有重叠
                if not (file_end < start_of_day or file_start > end_of_day):
                    matched_files.append((file, file_start, file_end))
        matched_files.sort(key=lambda x: x[1])
        return matched_files, start_of_day, end_of_day

    def process_data_for_date(self, target_date_str):
        ''' 处理指定日期的数据文件, 整理并保存为单独的CSV文件 '''
        matched_files, start_of_day, end_of_day = self.match_files_for_date(target_date_str)
        
        # 创建空的DataFrame存储结果
        daily_data = pd.DataFrame()
        
        for file_info in matched_files:
            file_name, _, _ = file_info
            file_path = os.path.join(self.data_dir, file_name)
            try:
                df = pd.read_csv(file_path)
                if 'DDATETIME' in df.columns:
                    df['DDATETIME'] = pd.to_datetime(df['DDATETIME'], format='%Y-%m-%d %H:%M:%S')
                    day_data = df[(df['DDATETIME'] >= start_of_day) & (df['DDATETIME'] <= end_of_day)]
                    daily_data = pd.concat([daily_data, day_data], ignore_index=True)
                else:
                    print(f"警告: 文件 {file_name} 中没有找到 'DDATETIME' 列")
            except Exception as e:
                print(f"读取文件 {file_name} 时出错: {str(e)}")
                continue
        
        if daily_data.empty:
            print(f"未找到 {target_date_str} 的数据")
            return
        
        # 按时间戳排序并去重（处理文件重叠部分）
        daily_data = daily_data.drop_duplicates(subset=['DDATETIME', 'GRIDID'], keep='first')
        daily_data.sort_values(by=['DDATETIME', 'GRIDID'], inplace=True)
        
        # 统计数据完整性, 需要返回给主程序
        completeness_info = analyze_data_completeness(daily_data, target_date_str, self.max_grid_id)

        # 保存结果
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        output_file = f"{self.save_dir}/{target_date_str}.csv"
        daily_data.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"已保存 {target_date_str} 的数据到 {output_file}")
        
        return {
            'date': target_date_str,
            'completeness_info': completeness_info
        }
    
    def concurrent_process_all_dates(self):
        """ 并发处理所有日期的数据 """
        start_date, end_date = self.get_all_date_range()
        
        print(f"数据时间范围: {start_date} 到 {end_date}, 数据总数: {(end_date - start_date).days + 1}")

        # 确保日期是datetime对象
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y%m%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y%m%d")
        
        # 生成需要处理的所有日期列表
        date_list = []
        current_date = start_date
        while current_date <= (end_date + timedelta(days=1)):
            date_list.append(current_date.strftime("%Y%m%d"))
            current_date += timedelta(days=1)
        
        # 使用线程池并发处理
        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 创建日期到future的映射
            future_to_date = {
                executor.submit(self.process_data_for_date, date_str): date_str
                for date_str in date_list
            }
            for future in concurrent.futures.as_completed(future_to_date):
                date_str = future_to_date[future]
                try:
                    results[date_str] = future.result()
                    if results[date_str] is not None:
                        self.completeness_stats[date_str] = results[date_str]['completeness_info']
                except Exception as exc:
                    print(f"处理日期 {date_str} 时发生错误: {exc}")
                    results[date_str] = None
        return results

    def run(self):
        """ 运行数据处理 """
        res = self.concurrent_process_all_dates()
        stats_file = os.path.join(self.save_dir, "0_completeness_stats.csv")
        self.save_completeness_stats_to_csv(stats_file)
        return res


class data_complettor:
    ''' data_compeleter 的数据补全类, 必须 import swifter '''
    def __init__(self, interpolate_configs):
        self.interpolate_configs = interpolate_configs
        self.spatial_threshold = 0.2
        self.temporal_threshold = 0.2
        # 处理矢量化列
        for vec_info in self.interpolate_configs['vector']['relationship']:
            if vec_info['type'] == 'scaler-direction' and vec_info['direct-sys'] == 'angle':
                scaler_col = vec_info['scaler']
                new_col_x = scaler_col + '_X'
                new_col_y = scaler_col + '_Y'
                # 按需进行
                self.interpolate_configs['scaler']['columns'] += [new_col_x, new_col_y]
                self.interpolate_configs['scaler']['methods'].update({
                    new_col_x: {'method': vec_info['method']},
                    new_col_y: {'method': vec_info['method']}
                    })
            else:
                raise ValueError(f'标量化方法未适配: {str(vec_info)}')

        self.columns_order = [
            "DDATETIME", "GRIDID", 
            "LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX", "LON_CENTER", "LAT_CENTER"
        ] + self.interpolate_configs['scaler']['columns']
    
    @staticmethod
    def calculate_air_density(df):
        ''' 计算空气密度 '''
        # 定义常数
        R_d = 287.05  # 干空气比气体常数, J/(kg·K)
        R_v = 461.495  # 水蒸气比气体常数, J/(kg·K)
        # 将温度转换为开尔文
        T_k = df['T'] + 273.15
        # 将气压转换为帕斯卡
        P_pa = df['SLP'] * 100
        # 计算饱和水汽压 (使用马格努斯公式)
        e_s = 6.112 * np.exp((17.62 * df['T']) / (df['T'] + 243.12))
        # 计算实际水汽压7
        e = (df['RHSFC'] / 100) * e_s
        # 将水汽压转换为帕斯卡
        e_pa = e * 100
        # 计算空气密度
        density = (P_pa / (R_d * T_k)) * (1 - (e_pa / P_pa) * (1 - R_d / R_v))
        return density

    def _interpolate_group_over_time(self, group):
        ''' 按组进行时间插值 '''
        group = group.sort_values('DDATETIME') # 确保组内按时间排序
        for col in self.interpolate_configs['scaler']['columns']:
            method = self.interpolate_configs['scaler']['methods'][col]
            if col in group.columns:
                nan_mask = group[col].isna()
                # 无需插值的情况
                if not nan_mask.any():
                    continue
                # 已知节点低于阈值，放弃插值
                if (nan_mask.sum()/len(nan_mask)) > self.spatial_threshold:
                    continue
                # 插值
                series = group[col]
                match method['method']:
                    case x if x in ['linear', 'cubic']:
                        interpolated_data = series.interpolate(method=method['method'])
                    case 'spline':
                        interpolated_data = series.interpolate(method='spline', order=method['order'])
                    case _:
                        raise ValueError('方法未定义')
                # 前后向传播
                interpolated_data = interpolated_data.ffill().bfill()
                # 替换
                group.loc[nan_mask, col] = interpolated_data[nan_mask]
                # 降雨不能为负
                if col.startswith('RAIN'):
                    group[col] = group[col].clip(lower=0)
        return group

    def _interpolate_group_over_space(self, group):
        ''' 按组进行空间插值 '''
        result_group = group.copy() # 
        # 提取坐标信息
        coords = result_group[['LON_CENTER', 'LAT_CENTER']].values
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        for col in self.interpolate_configs['scaler']['columns']:
            if col in result_group.columns:
                non_nan_mask = result_group[col].notna()
                nan_mask = result_group[col].isna()
                # 无需插值的情况
                if not nan_mask.any():
                    continue
                # 已知节点低于阈值，放弃插值
                if (nan_mask.sum()/len(nan_mask)) > self.temporal_threshold:
                    continue
                # 准备插值所需的数据
                known_coords = coords_scaled[non_nan_mask]
                known_values = result_group.loc[non_nan_mask, col].values
                unknown_coords = coords_scaled[nan_mask]
                # 使用径向基函数进行插值
                rbf = Rbf(known_coords[:, 0], known_coords[:, 1], known_values, function='multiquadric', smooth=1e-6, epsilon=2)
                interpolated_values = rbf(unknown_coords[:, 0], unknown_coords[:, 1])
                # 将插值结果填充回DataFrame
                result_group.loc[nan_mask, col] = interpolated_values
        return result_group
    
    def _spatialtemporal_interpolate(self, df, date_str):
        print(f'数据 {date_str} 正在进行空间插值')
        # 空间插值
        # df = df.groupby("DDATETIME").apply(self._interpolate_group_over_space).reset_index(drop=True) # For debug
        df = (
            df.swifter
            .progress_bar(False)
            .set_dask_scheduler('processes')
            .groupby("DDATETIME")
            .apply(self._interpolate_group_over_space)
            .reset_index(drop=True)
        )
        # 提前结束的情况
        if not df.isna().any().any():
            return df
        print(f'数据 {date_str} 正在进行时间插值')
        # 时间插值
        # df = df.groupby("GRIDID").apply(self._interpolate_group_over_time).reset_index(drop=True) # For debug
        df = (
            df.swifter
            .progress_bar(False)
            .set_dask_scheduler('processes')
            .groupby("GRIDID")
            .apply(self._interpolate_group_over_time)
            .reset_index(drop=True)
            )
        return df

    def vector_scalarization(self, df):
        for vec_info in self.interpolate_configs['vector']['relationship']:
            if vec_info['type'] == 'scaler-direction' and vec_info['direct-sys'] == 'angle':
                vector_col = vec_info['vector']
                scaler_col = vec_info['scaler']
                angles_rad = np.radians(df[vector_col])
                sin_vals = np.sin(angles_rad)
                cos_vals = np.cos(angles_rad)
                new_col_x = scaler_col + '_X'
                new_col_y = scaler_col + '_Y'
                df[new_col_x] = sin_vals * df[scaler_col]
                df[new_col_y] = cos_vals * df[scaler_col]
                df.drop([vector_col,scaler_col], axis=1, inplace=True)
            else:
                raise ValueError(f'标量化方法未适配: {str(vec_info)}')
        return df

    def interpolate_data_by_df(self, df, date_str):
        ''' 对 dataframe 的值进行补全 '''
        df_filled = df.copy()
        df_filled['DDATETIME'] = pd.to_datetime(df_filled['DDATETIME'])
        # 时空插值
        df_filled = self._spatialtemporal_interpolate(df_filled, date_str)
        # 检查完整性并尝试再次插值
        if df_filled.isna().any().any():
            # 再次尝试时空插值
            df_filled = self._spatialtemporal_interpolate(df_filled, date_str)
            flag = bool(not df_filled.isna().any().any())
        else:
            flag = True
        return df_filled, flag

    def add_additional_cols_by_date(self, df):
        # 添加空气密度信息
        df['AIR_DENSITY'] = self.calculate_air_density(df)
        if 'AIR_DENSITY' not in self.columns_order:
            self.columns_order.append('AIR_DENSITY')
        # 添加均值场信息
        result_df = df.copy()
        group_means = result_df.groupby('DDATETIME')[self.interpolate_configs['scaler']['columns']+['AIR_DENSITY']].mean().reset_index()
        # dataframe 重组
        combined_df = pd.concat([group_means, result_df], ignore_index=True)
        return combined_df

class data_completeness_processor(data_complettor):
    def __init__(self, 
        raw_data_directory='./daily_data_raw/',
        save_directory='./filled_data/',
        grid_info_path='./grid_info/grid_info.csv',
        stats_info_path='./daily_data_raw/0_completeness_stats.csv',
        interpolate_configs=None,
        max_grid_id=4231,
        completeness_threshold=0.95
    ):
        data_complettor.__init__(self, interpolate_configs)
        self.raw_data_directory = raw_data_directory
        self.save_directory = save_directory
        self.grid_info_path = grid_info_path
        self.stats_info_path = stats_info_path
        self.max_grid_id = max_grid_id
        self.completeness_threshold=completeness_threshold
        self._init_data_info()

    def _init_data_info(self):
        """ 初始化数据文件信息 """
        # 网格信息
        df_grid_info = pd.read_csv(self.grid_info_path)
        columns_map = {
            '格网ID（唯一）': 'GRIDID',
            '格网左下角经度（度）': 'LON_MIN',
            '格网左下角纬度（度）': 'LAT_MIN',
            '格网右上角经度（度）': 'LON_MAX',
            '格网右上角纬度（度）': 'LAT_MAX',
            '格网编码': 'GRID_CODE',
            '格网相对X坐标': 'X_IDX',
            '格网相对Y坐标': 'Y_IDX'
        }
        df_grid_info = df_grid_info.rename(columns=columns_map)
        df_grid_info = df_grid_info[list(columns_map.values())]
        df_grid_info.sort_values(by=['GRIDID'], inplace=True, ignore_index=True)
        df_grid_info = df_grid_info[list(columns_map.values())[:5]]
        self.df_grid_info = df_grid_info

        # 需要保留的列
        self.columns_to_keep = [
            'DDATETIME',    # 发布时间
            'GRIDID',       # 网格编号
            'T',            # 温度（摄氏度）
            'MAXTOFDAY',    # 日最高温度（摄氏度）
            'SLP',          # 气压（百帕）
            'RHSFC',        # 相对湿度（百分比）
            'V',            # 能见度（公里）
            'WSPD', 'WDIR', # 风速（米/秒），风向（度）
            'WD3SMAXDF', 'WD3SMAXDD',   # 极大风速（米/秒），极大风速风向（度）
            'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H',  # 1小时，2小时，3小时，6小时，24小时累计降雨量
        ]

        # 完整格点ID列表
        self.all_grids = df_grid_info['GRIDID'].unique()

        # 数据统计信息
        if self.stats_info_path is not None:
            stat_df = pd.read_csv(self.stats_info_path)
            stat_df['date'] = stat_df['date'].astype(str)
            self.stat_df = stat_df

    @staticmethod
    def _check_isolated(missing_times):
        """ 检查缺失的时间点中是否存在连续缺失的情况 """
        # 如果 missing_times 为空，直接返回 True
        if len(missing_times) == 0:
            return True
        sorted_times = pd.to_datetime(missing_times).sort_values()
        diffs = sorted_times.diff()
        has_continuous = any(diffs[1:] == pd.Timedelta('10 minutes'))
        return not has_continuous

    @staticmethod
    def _check_missing_times(date_str, df):
        """ 检查指定日期的数据中缺失的时间点 """
        start_time = datetime.strptime(date_str, "%Y-%m-%d")
        end_time = start_time + timedelta(days=1) - timedelta(minutes=10)
        time_series = pd.date_range(
            start=start_time,
            end=end_time,
            freq='10min'
        )
        time_series_str_series = pd.Series([ts.strftime("%Y-%m-%d %H:%M:%S") for ts in time_series])
        # 检查哪些时间点不在 df 中
        missing_mask = ~time_series_str_series.isin(df['DDATETIME'])
        missing_times = time_series_str_series[missing_mask]
        return missing_times

    @staticmethod
    def _get_full_times(date_str):
        # 完整时间
        start_time = datetime.strptime(date_str, "%Y-%m-%d")
        end_time = start_time + timedelta(days=1) - timedelta(minutes=10)
        time_series = pd.date_range(
            start=start_time,
            end=end_time,
            freq='10min'
        )
        time_strings = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in time_series]
        return time_strings
    
    
    def abnormal_correction(self, df, date_str):
        ''' 修正异常数据 '''
        # 气压
        position_mask = df['SLP'] == 0.0
        if position_mask.any().any():
            print(f"数据 {date_str} 存在异常气压值 {position_mask.sum()} 个")
            df['SLP'] = df['SLP'].replace(0, np.nan)
        # 湿度
        position_mask = df['RHSFC'] == 0.0
        if position_mask.any().any():
            print(f"数据 {date_str} 存在异常湿度值 {position_mask.sum()} 个")
            df['RHSFC'] = df['RHSFC'].replace(0, np.nan)
        return df

    def read_data_with_location_by_date(self, date_str):
        """ 读取指定日期的数据文件 """
        file_path = os.path.join(self.raw_data_directory, f"{date_str}.csv")
        df = pd.read_csv(file_path)
        df = df[self.columns_to_keep]
        df = self.abnormal_correction(df, date_str)

        # 获取日期
        dt = datetime.strptime(df['DDATETIME'].iloc[0], "%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%Y-%m-%d")
        # 检查缺失的时间点
        missing_times = self._check_missing_times(date_str, df)
        if not self._check_isolated(missing_times):
            pass
            print(f"警告: 日期 {date_str} 存在连续缺失的时间点: {missing_times.tolist()}")
            # raise ValueError("存在连续缺失的时间点")
        # 完整的时间-网格组合
        all_times = self._get_full_times(date_str)
        all_combinations = pd.MultiIndex.from_product(
            [all_times, self.all_grids],
            names=['DDATETIME', 'GRIDID']
        )
        df_full = pd.DataFrame(index=all_combinations).reset_index()
        df_merged = pd.merge(df_full, df, on=['DDATETIME', 'GRIDID'], how='left')
        # 合并网格信息
        df_with_location = pd.merge(df_merged, self.df_grid_info, on='GRIDID', how='left')
        # 计算网格中心坐标
        df_with_location['LON_CENTER'] = (df_with_location['LON_MIN'] + df_with_location['LON_MAX']) / 2
        df_with_location['LAT_CENTER'] = (df_with_location['LAT_MIN'] + df_with_location['LAT_MAX']) / 2
        return df_with_location, missing_times
    
    def process_single_date(self, date_str):
        # 读取
        df, _ = self.read_data_with_location_by_date(date_str)
        # 矢量数据标量化
        df = self.vector_scalarization(df)
        # 数据插值
        if df.isna().any().any():
            result_df, success = self.interpolate_data_by_df(df, date_str)
        else:
            result_df = df.copy()
        if not success:
            return None, date_str, False
        # 添加额外信息
        result_df = self.add_additional_cols_by_date(result_df)
        result_df = result_df[self.columns_order]
        filename = f"{date_str}.csv"
        # 保存
        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)
        result_df.to_csv(self.save_directory+filename, index=False, encoding='utf-8-sig')
        print(f'日期 {date_str} 结果已保存')

        return result_df, date_str, True

    def process_all_data_concurrently(self, n_processes=8):
        ''' 并行处理所有日期数据文件 '''
        # 获取所有符合要求的数据
        pattern = os.path.join(self.raw_data_directory, "[0-9]"*8 + ".csv")
        file_paths = glob.glob(pattern)
        file_paths.sort()
        date_list = []
        ignore_list = []
        for path in file_paths:
            date_str = os.path.basename(path).replace('.csv', '')
            completeness_rate = safe_lookup(self.stat_df, 'date', date_str, 'completeness_ratio')
            if completeness_rate >= self.completeness_threshold:
                date_list.append(date_str)
            else:
                ignore_list.append(date_str)
        date_list.sort()
        # 并行处理所有符合完整性条件的数据并将结果保存在目标目录
        print(f"使用 {n_processes} 个进程处理 {len(date_list)} 个日期")
        with Pool(processes=n_processes) as pool:
            for result, date_str, success in tqdm(
                pool.imap_unordered(self.process_single_date, date_list),
                total=len(date_list),
                desc="当前进度"
            ):
                if not success:
                    print(f'数据 {date_str} 处理失败，已放弃')