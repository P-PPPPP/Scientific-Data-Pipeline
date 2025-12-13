import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # pip install tqdm

class CSVToBinConverter:
    def __init__(self, data_dir, target_dir):
        self.data_dir = Path(data_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 原始读取列
        self.keep_columns = [
            'DDATETIME', 'LON_CENTER', 'LAT_CENTER',
            'u10', 'v10', 'd2m', 't2m', 'msl', 'sp',
            'i10fg', 'cape', 'lcc', 'tcc', 'blh', 'tcwv',
            'skt', 'ssrd', 'tp', 'z', 'lsm'
        ]
        
        # 2. 最终输出特征顺序
        self.final_features = [
            'u10', 'v10', 'ws',            # Wind
            't2m', 'd2m', 'rh',            # Temp & Hum
            'sp', 'msl',                   # Pressure
            'i10fg', 'cape',               # Extremes
            'tp', 'ssrd',                  # Drivers
            'lcc', 'tcc',                  # Clouds
            'blh', 'tcwv', 'skt',          # Env State
            'z', 'lsm'                     # Static
        ]
        
        # 3. Log1p 变换列
        self.log_transform_columns = ['ssrd', 'tp', 'cape', 'i10fg']
        
        self.num_grids = 3276
        self.time_steps_per_day = 24
        
    def calculate_derived_features(self, df):
        """计算物理衍生变量：RH 和 WS"""
        # --- RH 计算 ---
        t_c = df['t2m'] - 273.15
        td_c = df['d2m'] - 273.15
        
        # Magnus 公式
        es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
        e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
        
        df['rh'] = (e / es).clip(0, 1)

        # --- WS 计算 ---
        df['ws'] = np.sqrt(df['u10']**2 + df['v10']**2)
        
        return df

    def process_units_and_log(self, df):
        """物理单位修正 和 Log1p 变换"""
        
        # 温度: K -> C
        for col in ['t2m', 'd2m', 'skt']:
            if col in df.columns:
                df[col] = df[col] - 273.15

        # 气压: Pa -> Bar (100 kPa)
        for col in ['sp', 'msl']:
            if col in df.columns:
                df[col] = df[col] / 100000.0

        # 高度: m / m2s2 -> km
        if 'z' in df.columns:
            df['z'] = df['z'] / 9806.65 # Geopotential -> Height (km)
        
        if 'blh' in df.columns:
            df['blh'] = df['blh'] / 1000.0 # m -> km

        # 百分比截断
        for col in ['lcc', 'tcc']:
            if col in df.columns:
                df[col] = df[col].clip(0, 1)

        # 降水: m -> mm
        if 'tp' in df.columns:
            df['tp'] = (df['tp'] * 1000).clip(lower=0)

        # 辐射: J/m2 -> W/m2
        if 'ssrd' in df.columns:
             df['ssrd'] = (df['ssrd'] / 3600.0).clip(lower=0)

        # Log1p 变换
        for col in self.log_transform_columns:
            if col in df.columns:
                df[col] = np.log1p(df[col])
                
        return df

    def _process_wrapper(self, csv_file):
        """
        多进程包装器：处理单个文件并返回状态和结果，不直接修改类属性
        """
        try:
            bin_file = self.target_dir / f"{csv_file.stem}.bin"
            time_strs, grid_coords = self.process_single_file(csv_file, bin_file)
            
            return {
                "status": "success",
                "stem": csv_file.stem,
                "time_strs": time_strs,
                "grid_coords": grid_coords
            }
        except Exception as e:
            return {
                "status": "error",
                "file": csv_file.name,
                "error_msg": str(e)
            }

    def process_single_file(self, csv_file, bin_file):
        """核心处理逻辑"""
        # [性能建议] 如果安装了 pyarrow，这里改成 engine='pyarrow' 速度会快 5-10 倍
        # df = pd.read_csv(csv_file, engine='pyarrow') 
        df = pd.read_csv(csv_file)
        
        # 1. 过滤列
        existing_cols = [c for c in self.keep_columns if c in df.columns]
        df = df[existing_cols]
        
        # 2. 特征工程
        df = self.calculate_derived_features(df)
        
        # 3. 单位与Log
        df = self.process_units_and_log(df)
        
        # 4. 对齐网格
        grid_data = df.copy()
        expected_len = self.time_steps_per_day * self.num_grids
        
        if len(grid_data) != expected_len:
             if len(grid_data) > expected_len:
                 grid_data = grid_data.iloc[-expected_len:]
             else:
                 raise ValueError(f"Length mismatch: {len(grid_data)} < {expected_len}")
        
        # 动态生成 GRIDID (确保所有文件排序逻辑一致)
        unique_grids_coords = grid_data[['LON_CENTER', 'LAT_CENTER']].drop_duplicates()
        # 严格按照 Lat, Lon 排序以保证 ID 稳定
        unique_grids_coords = unique_grids_coords.sort_values(by=['LAT_CENTER', 'LON_CENTER']).reset_index(drop=True)
        unique_grids_coords['GRIDID'] = unique_grids_coords.index

        grid_data = grid_data.merge(unique_grids_coords, on=['LON_CENTER', 'LAT_CENTER'], how='left')
        grid_data = grid_data.sort_values(['DDATETIME', 'GRIDID'])
        
        # 5. 提取特征
        for col in self.final_features:
            if col not in grid_data.columns:
                grid_data[col] = 0.0
                
        # 提取并转为 float32
        weather_features = grid_data[self.final_features].values.astype(np.float32)
        
        data_3d = weather_features.reshape(
            self.time_steps_per_day, self.num_grids, len(self.final_features)
        )
        
        # 提取坐标
        coords = grid_data[['LON_CENTER', 'LAT_CENTER']].values
        # 只需要第一个时间步的坐标即可 (所有时间步坐标相同)
        grid_coords = coords[:self.num_grids].astype(np.float32)
        
        # 提取时间
        unique_times = grid_data['DDATETIME'].unique()
        time_strs = unique_times.tolist()
        
        # 写入
        with open(bin_file, 'wb') as f:            
            data_3d.tofile(f)
        
        return time_strs, grid_coords

    def process_all_files(self, max_workers=None):
        """多进程入口"""
        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        
        global_time_strs = {}
        first_grid_coords = None 
        
        # 自动决定进程数，保留 2 个核心给系统
        import os
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 2)

        print(f"Starting parallel processing with {max_workers} workers on {len(csv_files)} files...")
        start_time = time.time()

        

        # 启动进程池
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            futures = [executor.submit(self._process_wrapper, f) for f in csv_files]
            
            # 使用 tqdm 监控进度
            for future in tqdm(as_completed(futures), total=len(csv_files), unit="file"):
                result = future.result()
                
                if result['status'] == 'success':
                    # 收集元数据
                    global_time_strs[result['stem']] = result['time_strs']
                    
                    # 只保存第一份坐标数据作为全局坐标
                    if first_grid_coords is None:
                        first_grid_coords = result['grid_coords']
                else:
                    print(f"\n[Error] File {result['file']} failed: {result['error_msg']}")

        # 排序时间元数据 (因为多进程返回顺序是乱的)
        sorted_keys = sorted(global_time_strs.keys())
        sorted_global_times = {k: global_time_strs[k] for k in sorted_keys}

        # 保存全局元数据
        if first_grid_coords is not None:
            self.save_global_metadata(sorted_global_times, first_grid_coords)
            print(f"\nDone! Total time: {time.time() - start_time:.2f}s")
        else:
            print("\nFailed to process any files successfully.")
    
    def save_global_metadata(self, global_time_strs, grid_coords):
        """保存全局元数据"""
        print("Saving global metadata...")
        global_metadata = {
            'num_features': len(self.final_features),
            'num_grids': self.num_grids,
            'time_steps_per_day': self.time_steps_per_day,
            'feature_names': self.final_features,
            'coord_names': ['LON_CENTER', 'LAT_CENTER'],
            'dtype': 'float32',
            'log_transformed_features': self.log_transform_columns,
            'units_note': {
                'temperature': 'Celsius',
                'pressure': '100kPa (Bar)',
                'height_z_blh': 'Kilometers (km)',
                'percentage': '0-1 decimal',
                'precip': 'mm',
                'radiation': 'W/m2'
            }
        }

        with open(self.target_dir / 'metadata.json', 'w') as f:
            json.dump(global_metadata, f, indent=2)

        with open(self.target_dir / 'date_data.json', 'w') as f:
            json.dump(global_time_strs, f, indent=2)

        np.save(self.target_dir / 'coords_data.npy', grid_coords)

# 使用示例
if __name__ == "__main__":
    # 请根据实际路径修改
    data_dir = '/mnt/drive1/pengpeng/storage/era5/era5_daily_data_global'
    target_dir = '/mnt/drive1/pengpeng/storage/era5/era5_bin_data_global'
    
    # 初始化并运行
    converter = CSVToBinConverter(data_dir, target_dir)
    # 可以手动指定 max_workers，例如 max_workers=10
    converter.process_all_files()