import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


class CSVToBinConverter:
    def __init__(self, data_dir, target_dir):
        self.data_dir = Path(data_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 原始列
        self.keep_columns = [
            'DDATETIME',
            'GRIDID',
            'LON_CENTER', 'LAT_CENTER', # 经纬度
            'T',  # 温度，摄氏度
            'MAXTOFDAY', # 日最高温度，摄氏度
            'SLP', # 气压，百帕
            'RHSFC', # 相对湿度 [0, 100]
            'V', # 能见度 公里
            'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H', # 1小时到24小时累计降水量，毫米
            'WSPD_X', 'WSPD_Y',  # 风速分量，米/秒
            'WD3SMAXDF_X', 'WD3SMAXDF_Y',  # 极大风速 米/秒， 极大风向 度
            'AIR_DENSITY' # 空气密度
        ]

        # 最终输出特征顺序
        self.weather_columns = [
            'T', 'MAXTOFDAY', 'SLP', 'AIR_DENSITY',
            'RHSFC',
            'WSPD_X', 'WSPD_Y', 'WSPD',
            'WD3SMAXDF_X', 'WD3SMAXDF_Y', 'GUST',
            'V',
            'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H'
        ]
        
        # 需要 Log1p 变换的列
        self.log_transform_columns = [
            'V', 'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H'
        ]
        
        self.num_grids = 4232
        self.time_steps_per_day = 144
        
    def calculate_derived_features(self, df):
        """计算衍生变量 和 单位换算"""
        # 气压: hPa -> kPa
        if 'SLP' in df.columns:
            df['SLP'] = df['SLP'] / 10.0
            
        # 平均风速合成
        df['WSPD'] = np.sqrt(df['WSPD_X']**2 + df['WSPD_Y']**2)
        
        # 极大风速合成
        df['GUST'] = np.sqrt(df['WD3SMAXDF_X']**2 + df['WD3SMAXDF_Y']**2)
        
        return df

    def process_log_transform(self, df):
        """对长尾分布数据应用 Log1p"""
        for col in self.log_transform_columns:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df

    def _process_wrapper(self, csv_file):
        """
        包装器函数，用于多进程调用。
        它处理单个文件并返回元数据，不修改类属性。
        """
        try:
            bin_file = self.target_dir / f"{csv_file.stem}.bin"
            time_strs, grid_coords = self.process_single_file(csv_file, bin_file)
            return {
                "status": "success",
                "stem": csv_file.stem,
                "time_strs": time_strs,
                "grid_coords": grid_coords,
                "file": csv_file.name
            }
        except Exception as e:
            return {
                "status": "error",
                "file": csv_file.name,
                "error": str(e)
            }

    def process_single_file(self, csv_file, bin_file):
        """处理单个CSV文件的核心逻辑"""
        # df = pd.read_csv(csv_file, engine='pyarrow') 
        df = pd.read_csv(csv_file)
        
        # 丢弃前144行
        grid_data = df.iloc[144:].copy()
        
        # 过滤列
        grid_data = grid_data[self.keep_columns]
        
        # 特征工程
        grid_data = self.calculate_derived_features(grid_data)
        
        # Log 变换
        grid_data = self.process_log_transform(grid_data)
        
        # 验证长度
        expected_len = self.time_steps_per_day * self.num_grids
        if len(grid_data) != expected_len:
             raise ValueError(f"数据长度不匹配: {len(grid_data)} != {expected_len}")
        
        # 排序
        grid_data = grid_data.sort_values(['DDATETIME', 'GRIDID'])
        
        # 提取时间
        unique_times = grid_data['DDATETIME'].unique()
        if len(unique_times) != self.time_steps_per_day:
            raise ValueError(f"时间步错误: {len(unique_times)} != {self.time_steps_per_day}")

        # 填充缺失列
        for col in self.weather_columns:
            if col not in grid_data.columns:
                 grid_data[col] = 0.0
                 
        # 提取矩阵并 reshape
        # 使用 float32 节省内存和磁盘IO
        weather_features = grid_data[self.weather_columns].values.astype(np.float32)
        
        data_3d = weather_features.reshape(
            self.time_steps_per_day, self.num_grids, len(self.weather_columns)
        )
        
        # 提取坐标
        coords = grid_data[['LON_CENTER', 'LAT_CENTER']].values[:self.num_grids].astype(np.float32)
        
        time_strs = unique_times.tolist()
        
        # 写入二进制
        with open(bin_file, 'wb') as f:            
            data_3d.tofile(f)
        
        return time_strs, coords

    def process_all_files(self, max_workers=None):
        """使用多进程处理所有文件"""
        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        
        global_time_strs = {}
        first_grid_coords = None
        
        # 如果不指定 workers，默认为 CPU 核心数
        if max_workers is None:
            import os
            # 这里的逻辑是保留一点资源给系统，避免死机
            max_workers = max(1, os.cpu_count() - 2) 

        print(f"Starting processing {len(csv_files)} files with {max_workers} processes...")
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(self._process_wrapper, f) for f in csv_files]
            
            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(futures), total=len(csv_files), unit="file"):
                result = future.result()
                
                if result['status'] == 'success':
                    # 收集元数据
                    global_time_strs[result['stem']] = result['time_strs']
                    
                    if first_grid_coords is None:
                        first_grid_coords = result['grid_coords']
                else:
                    print(f"\nError processing {result['file']}: {result['error']}")

        # 排序并保存全局元数据
        # 确保 global_time_strs 按照文件名排序，因为多进程返回顺序是随机的
        sorted_keys = sorted(global_time_strs.keys())
        sorted_global_time = {k: global_time_strs[k] for k in sorted_keys}

        if first_grid_coords is not None:
            self.save_global_metadata(sorted_global_time, first_grid_coords)
            print(f"\n处理完成！耗时: {time.time() - start_time:.2f} 秒")
        else:
            print("\n警告：没有成功处理任何文件，未保存元数据。")

    def save_global_metadata(self, global_time_strs, grid_coords):
        """保存全局元数据"""
        print("Saving metadata...")
        global_metadata = {
            'num_features': len(self.weather_columns),
            'num_grids': self.num_grids,
            'time_steps_per_day': self.time_steps_per_day,
            'feature_names': self.weather_columns,
            'coord_names': ['LON_CENTER', 'LAT_CENTER'],
            'dtype': 'float32',
            'log_transformed_features': self.log_transform_columns,
            'units_note': {
                'temperature': 'Celsius',
                'pressure': '100kPa (Bar)',
                'humidity': '0-1 Decimal',
                'visibility': 'km',
                'wind': 'm/s',
                'rain': 'mm'
            }
        }
        
        with open(self.target_dir / 'metadata.json', 'w') as f:
            json.dump(global_metadata, f, indent=2)

        with open(self.target_dir / 'date_data.json', 'w') as f:
            json.dump(global_time_strs, f, indent=2)

        np.save(self.target_dir / 'coords_data.npy', grid_coords)

if __name__ == "__main__":
    # 路径配置
    data_dir = './storage/sz_weather/csv_data'
    test_dir = './storage/sz_weather/bin_data'
    
    # 可以在这里手动指定进程数，例如 max_workers=8
    converter = CSVToBinConverter(data_dir, test_dir)
    converter.process_all_files(max_workers=32)