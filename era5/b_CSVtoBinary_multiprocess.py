import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


class CSVToBinConverter:
    def __init__(self, 
                 data_dir, 
                 target_dir, 
                 keep_columns, 
                 final_features, 
                 log_transform_columns,
                 num_grids,
                 time_steps_per_day=24):
        """
        初始化转换器，参数由外部传入
        """
        self.data_dir = Path(data_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 参数配置
        self.keep_columns = keep_columns
        self.final_features = final_features
        self.log_transform_columns = log_transform_columns
        self.num_grids = num_grids
        self.time_steps_per_day = time_steps_per_day
        
    def calculate_derived_features(self, df):
        """计算物理衍生变量：RH 和 WS"""
        # --- RH 计算 ---
        if 't2m' in df.columns and 'd2m' in df.columns:
            t_c = df['t2m'] - 273.15
            td_c = df['d2m'] - 273.15
            
            # Magnus 公式
            es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
            e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
            
            df['rh'] = (e / es).clip(0, 1) * 100

        # --- WS 计算 ---
        if 'u10' in df.columns and 'v10' in df.columns:
            df['ws'] = np.sqrt(df['u10']**2 + df['v10']**2)
        
        return df

    def process_units_and_log(self, df):
        """物理单位修正 和 Log1p 变换"""
        
        # 温度: K -> C
        for col in ['t2m', 'd2m', 'skt']:
            if col in df.columns:
                df[col] = df[col] - 273.15

        # 气压: Pa -> bar
        for col in ['sp', 'msl']:
            if col in df.columns:
                df[col] = df[col] / 1000.0

        # 高度: m / m2s2 -> km
        if 'z' in df.columns:
            df['z'] = df['z'] / 9806.65 # Geopotential -> Height (km)
        
        if 'blh' in df.columns:
            df['blh'] = df['blh'] # m -> km

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

        # Log1p 变换 (基于配置列表)
        for col in self.log_transform_columns:
            if col in df.columns:
                df[col] = np.log1p(df[col])
                
        return df

    def _process_wrapper(self, csv_file):
        """多进程包装器"""
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
        # 如果追求极致速度，可尝试 engine='pyarrow'
        df = pd.read_csv(csv_file)
        
        # 过滤列
        existing_cols = [c for c in self.keep_columns if c in df.columns]
        df = df[existing_cols]
        
        # 特征工程
        df = self.calculate_derived_features(df)
        
        # 单位与Log
        df = self.process_units_and_log(df)
        
        # 对齐网格
        grid_data = df.copy()
        expected_len = self.time_steps_per_day * self.num_grids
        
        if len(grid_data) != expected_len:
             if len(grid_data) > expected_len:
                 grid_data = grid_data.iloc[-expected_len:]
             else:
                 raise ValueError(f"Length mismatch: {len(grid_data)} < {expected_len}")
        
        # 动态生成 GRIDID
        unique_grids_coords = grid_data[['LON_CENTER', 'LAT_CENTER']].drop_duplicates()
        unique_grids_coords = unique_grids_coords.sort_values(by=['LAT_CENTER', 'LON_CENTER']).reset_index(drop=True)
        unique_grids_coords['GRIDID'] = unique_grids_coords.index

        grid_data = grid_data.merge(unique_grids_coords, on=['LON_CENTER', 'LAT_CENTER'], how='left')
        grid_data = grid_data.sort_values(['DDATETIME', 'GRIDID'])
        
        # 提取特征
        for col in self.final_features:
            if col not in grid_data.columns:
                # 如果某个配置的特征计算失败或缺失，填0兜底
                grid_data[col] = 0.0
                
        # 提取并转为 float32
        weather_features = grid_data[self.final_features].values.astype(np.float32)
        
        # Reshape: [Time, Grid, Feature]
        data_3d = weather_features.reshape(
            self.time_steps_per_day, self.num_grids, len(self.final_features)
        )
        
        # 提取坐标 (只取第一个时间步)
        coords = grid_data[['LON_CENTER', 'LAT_CENTER']].values
        grid_coords = coords[:self.num_grids].astype(np.float32)
        
        # 提取时间
        unique_times = grid_data['DDATETIME'].unique()
        time_strs = unique_times.tolist()
        
        # 写入 Bin
        with open(bin_file, 'wb') as f:            
            data_3d.tofile(f)
        
        return time_strs, grid_coords

    def process_all_files(self, max_workers=None):
        """多进程入口"""
        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return

        global_time_strs = {}
        first_grid_coords = None 
        
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 2)

        print(f"Starting processing {len(csv_files)} files with {max_workers} workers...")
        print(f"   Input: {self.data_dir}")
        print(f"   Output: {self.target_dir}")
        print(f"   Grid Size: {self.num_grids}, Features: {len(self.final_features)}")
        
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_wrapper, f) for f in csv_files]
            
            for future in tqdm(as_completed(futures), total=len(csv_files), unit="file"):
                result = future.result()
                
                if result['status'] == 'success':
                    global_time_strs[result['stem']] = result['time_strs']
                    if first_grid_coords is None:
                        first_grid_coords = result['grid_coords']
                else:
                    print(f"\n[Error] File {result['file']} failed: {result['error_msg']}")

        # 整理元数据
        sorted_keys = sorted(global_time_strs.keys())
        sorted_global_times = {k: global_time_strs[k] for k in sorted_keys}

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


# ================= 配置与执行区域 =================
if __name__ == "__main__":
    
    # 配置
    INPUT_DIR = '~/storage/era5/era5_daily_data_global'
    OUTPUT_DIR = '~/storage/era5/bin_data_global'
    AREA = None
    GRID = [5, 5]

    # INPUT_DIR = '~/storage/era5/era5_daily_data_cn'
    # OUTPUT_DIR = '~/storage/era5/bin_data_cn'
    # AREA = [54, 73, 3, 135]
    # GRID = [1, 1]

    # --- 计算 NUM_GRIDS ---
    lat_step, lon_step = GRID

    if AREA is None:
        # 全球
        num_lat = int(round(180 / lat_step)) + 1
        num_lon = int(round(360 / lon_step))
        print(f"Mode: Global (Default)")
    else:
        # 指定区域
        north, west, south, east = AREA
        num_lat = int(round((north - south) / lat_step)) + 1
        # 计算经度 (需判断是否横跨整个地球)
        if east < west:
            lon_span = (east + 360) - west
        else:
            lon_span = east - west
        # 如果跨度非常接近 360 度，视为全球模式，不 +1
        if abs(lon_span - 360) < 1e-6:
            num_lon = int(round(lon_span / lon_step))
            print(f"🌍 Mode: Explicit Global (360° detected)")
        else:
            # 局部区域，首尾不相连 -> 需要 +1
            num_lon = int(round(lon_span / lon_step)) + 1
            print(f"🗺️  Mode: Regional Crop")

    # 总网格数
    NUM_GRIDS = num_lat * num_lon
    
    print(f"Grid Calculation Info:")
    print(f"   Latitude Points : {num_lat}")
    print(f"   Longitude Points: {num_lon}")
    print(f"   Total Grids     : {num_lat} * {num_lon} = {NUM_GRIDS}")
    # -----------------------

    TIME_STEPS = 24

    RAW_COLUMNS = [
        'DDATETIME', 'LON_CENTER', 'LAT_CENTER',
        # --- 风场基础 (Wind Basis) ---
        'u10', 'v10',       # 10m 纬向/经向风分量：用于合成全风速 (WS) 和确定风向
        # --- 温湿基础 (Temp & Hum Basis) ---
        'd2m',              # 2m 露点温度：计算相对湿度 (RH) 和空气密度的关键
        't2m',              # 2m 气温：直接对应 AWS 观测，回归分析基础 (后续转 °C)
        # --- 气压基础 (Pressure Basis) ---
        'msl',              # 海平面气压：用于消除海拔差异的大尺度背景气压 (后续转 Bar)
        'sp',               # 地表气压：基于网格平均海拔的气压，需配合 z 进行垂直订正 (后续转 Bar)
        # --- 极端与对流 (Extremes & Convection) ---
        'i10fg',            # 瞬时10m阵风：捕捉台风/强对流极值 (长尾分布 -> Log)
        'cape',             # 对流有效位能：预测雷暴/大风潜势，物理合理性判据 (长尾分布 -> Log)
        # --- 云与能见度 (Clouds & Visibility) ---
        'lcc',              # 低云量：关联“回南天”和大雾，填补能见度的关键特征
        'tcc',              # 总云量：影响辐射降温和白天的升温幅度 (0-1)
        # --- 边界层与水汽 (Boundary Layer & Vapor) ---
        'blh',              # 边界层高度：决定污染物垂直扩散，关联能见度
        'tcwv',             # 整层水汽 (kg/m2 或 mm)：数值范围约 1-70。若全部凝结对应的水深。暴雨发生的绝对水汽条件。
        # --- 地表热力与辐射 (Surface Thermal & Radiation) ---
        'skt',              # 肤温 (Skin Temp)：对辐射响应快，用于监测城市热岛效应 (后续转 °C)
        'ssrd',             # 地表短波辐射：气温上升的直接驱动力 (J/m2 -> W/m2 -> Log)
        # --- 降水 (Precipitation) ---
        'tp',               # 总降水量：直接对应 AWS 雨量计 (m -> mm -> Log)
        # --- 静态地理信息 (Geo-Static) ---
        'z',                # 位势：地势高度 (z/9.8)，用于气温/气压的垂直递减率订正
        'lsm'               # 海陆掩码：区分深圳滨海特征 (陆地/海洋热力差异)
    ]
        
    # 最终输出到 .bin 文件的特征列表
    # 模型输入通道顺序 (Channel Order)
    FINAL_FEATURES = [
        # Group 1: 风 (Wind)
        'u10', 'v10',       # 风矢量分量
        'ws',               # [衍生] 合成风速 (Wind Speed)
        # Group 2: 温湿 (Temperature & Humidity)
        't2m',              # 气温 (Celsius)
        'd2m',              # 露点 (Celsius)
        'rh',               # [衍生] 相对湿度 (0-1, Magnus公式计算)
        # Group 3: 气压 (Pressure)
        'sp',               # 地表压 (Bar / 100kPa)
        'msl',              # 海平压 (Bar / 100kPa)
        # Group 4: 极值与潜势 (Extremes & Potential)
        'i10fg',            # 阵风 (Log)
        'cape',             # 对流潜势 (Log)
        # Group 5: 水与光 (Water & Light - Drivers)
        'tp',               # 降水 (mm + Log)
        'ssrd',             # 辐射 (W/m2 + Log)
        # Group 6: 云 (Clouds)
        'lcc',              # 低云 (0-1)
        'tcc',              # 总云 (0-1)
        # Group 7: 环境状态 (Environmental State)
        'blh',              # 边界层高度 (km) - 影响扩散
        'tcwv',             # 整层水汽
        'skt',              # 肤温 (Celsius)
        # Group 8: 地理静态 (Static Geography)
        'z',                # 地形高度 (km) - 垂直订正核心
        'lsm'               # 海陆掩码
    ]

    LOG_TRANSFORM_COLS = ['ssrd', 'tp', 'cape', 'i10fg']

    converter = CSVToBinConverter(
        data_dir=INPUT_DIR,
        target_dir=OUTPUT_DIR,
        keep_columns=RAW_COLUMNS,
        final_features=FINAL_FEATURES,
        log_transform_columns=LOG_TRANSFORM_COLS,
        num_grids=NUM_GRIDS,
        time_steps_per_day=TIME_STEPS
    )

    converter.process_all_files(max_workers=1)