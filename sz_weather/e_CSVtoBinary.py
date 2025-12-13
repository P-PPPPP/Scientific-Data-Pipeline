import numpy as np
import pandas as pd
from pathlib import Path
import json

class CSVToBinConverter:
    def __init__(self, data_dir, target_dir):
        self.data_dir = Path(data_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 原始列 (CSV中读取)
        # 定义要保留的列，包含物理含义解释
        self.keep_columns = [
            'DDATETIME', # 时间戳
            'GRIDID',    # 网格标识
            'LON_CENTER', 'LAT_CENTER', # 经纬度
            
            # --- 温度 (Temperature) ---
            # 直接对应 ERA5 的 t2m
            'T',          # 2m 气温 (摄氏度 °C)。基础热力变量。
            'MAXTOFDAY',  # 日最高气温 (摄氏度 °C)。反映白天加热强度的极值，对高温预警至关重要。
            
            # --- 气压 (Pressure) ---
            # 对应 ERA5 的 msl。
            # 注意：AWS 的 SLP 通常指校正到海平面的气压，用于消除海拔影响。
            'SLP',        # 海平面气压 (hPa -> Bar)。大尺度天气背景（高压脊/低压槽）的指示器。
            
            # --- 湿度与能见度 (Moisture & Visibility) ---
            # 对应 ERA5 的 rh 和 lcc/blh 组合
            'RHSFC',      # 2m 相对湿度 (% -> 0-1)。反映空气饱和程度，降水和成雾的关键条件。
            'V',          # 水平能见度 (km)。受湿度、降水和气溶胶共同影响。长尾分布 -> Log变换。
            
            # --- 降水 (Precipitation) ---
            # 对应 ERA5 的 tp
            'RAIN01H',    # 过去1小时累计雨量 (mm)。短时强降水核心指标。
            'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H', # 不同时间尺度的累计雨量，用于捕捉降水持续性。
            
            # --- 风 (Wind) ---
            # 对应 ERA5 的 u10/v10
            'WSPD_X',     # 纬向风速 (m/s)。西风为正。
            'WSPD_Y',     # 经向风速 (m/s)。南风为正。
            'WD3SMAXDF_X',# 极大风速X分量 (m/s)。
            'WD3SMAXDF_Y',# 极大风速Y分量 (m/s)。
            
            # --- 空气密度 (Density) ---
            # 这是一个物理衍生量，P = rho * R * T，通常用于风能计算或污染物扩散修正
            'AIR_DENSITY' # 空气密度 (kg/m3)。
        ]
        
        # 2. 最终输出特征顺序
        # 模型输入通道顺序 (Channel Order)
        self.weather_columns = [
            # Group 1: 基础热力与动力
            'T',            # 气温 (°C)
            'MAXTOFDAY',    # 最高温 (°C)
            'SLP',          # 海平面气压 (Bar / 100kPa)
            'AIR_DENSITY',  # 密度
            
            # Group 2: 湿度
            'RHSFC',        # 相对湿度 (0-1)
            
            # Group 3: 风场 (平均风 + 极大风)
            'WSPD_X', 'WSPD_Y', 'WSPD',       # 平均风：分量 + 合成强度
            'WD3SMAXDF_X', 'WD3SMAXDF_Y', 'GUST', # 极大风(阵风)：分量 + 合成强度
            
            # Group 4: 能见度 (Log)
            'V',            # 能见度 (km)
            
            # Group 5: 降水梯队 (Log)
            'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H'
        ]
        
        # 3. 需要 Log1p 变换的列 (长尾分布)
        self.log_transform_columns = [
            'V', # 能见度：关注低能见度端，且数值跨度大(0-30km)
            'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H'
        ]
        
        self.num_grids = 4232
        self.time_steps_per_day = 144
        
    def calculate_derived_features(self, df):
        """计算衍生变量 和 单位换算"""
        
        # 1. 气压单位转换: hPa -> Bar (100 kPa)
        # 1 Bar = 1000 hPa
        if 'SLP' in df.columns:
            df['SLP'] = df['SLP'] / 1000.0
            
        # 2. 湿度单位转换: % -> Decimal (0-1)
        if 'RHSFC' in df.columns:
            df['RHSFC'] = df['RHSFC'] / 100.0
            df['RHSFC'] = df['RHSFC'].clip(0, 1) # 严格限制在 0-1

        # 3. 平均风速合成
        df['WSPD'] = np.sqrt(df['WSPD_X']**2 + df['WSPD_Y']**2)
        
        # 4. 极大风速(阵风)合成
        df['GUST'] = np.sqrt(df['WD3SMAXDF_X']**2 + df['WD3SMAXDF_Y']**2)
        
        return df

    def process_log_transform(self, df):
        """对长尾分布数据应用 Log1p"""
        for col in self.log_transform_columns:
            if col in df.columns:
                # 物理约束：确保非负
                df[col] = df[col].clip(lower=0)
                # Log1p 变换: log(x+1)
                df[col] = np.log1p(df[col])
        return df

    def process_all_files(self):
        """处理所有CSV文件"""
        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        global_time_strs = {}
        first_grid_coords = None

        for i, csv_file in enumerate(csv_files):
            print(f"Processing {csv_file.name} ({i+1}/{len(csv_files)})")
            bin_file = self.target_dir / f"{csv_file.stem}.bin"
            
            time_strs, grid_coords = self.process_single_file(csv_file, bin_file)
            global_time_strs[csv_file.stem] = time_strs
            
            if first_grid_coords is None:
                first_grid_coords = grid_coords
        
        # 保存全局元数据
        self.save_global_metadata(global_time_strs, first_grid_coords)
    
    def process_single_file(self, csv_file, bin_file):
        """处理单个CSV文件"""
        # 1. 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 2. 丢弃前144行全局平均数据
        grid_data = df.iloc[144:].copy()
        
        # 3. 过滤列
        grid_data = grid_data[self.keep_columns]
        
        # 4. 特征工程 (单位换算 + 风速合成)
        grid_data = self.calculate_derived_features(grid_data)
        
        # 5. Log 变换
        grid_data = self.process_log_transform(grid_data)
        
        # 6. 验证数据完整性
        expected_len = self.time_steps_per_day * self.num_grids
        
        if len(grid_data) != expected_len:
             raise ValueError(f"数据长度不匹配: 实际 {len(grid_data)} != 预期 {expected_len} "
                              f"(文件: {csv_file.name})")
        
        # 7. 排序
        grid_data = grid_data.sort_values(['DDATETIME', 'GRIDID'])
        
        # 提取时间戳
        unique_times = grid_data['DDATETIME'].unique()
        if len(unique_times) != self.time_steps_per_day:
            raise ValueError(f"时间步数量错误: {len(unique_times)} (预期 {self.time_steps_per_day})")

        # 8. 填充缺失列
        for col in self.weather_columns:
            if col not in grid_data.columns:
                 grid_data[col] = 0.0
                 
        # 9. 提取数值矩阵
        weather_features = grid_data[self.weather_columns].values
        
        # 10. Reshape
        data_3d = weather_features.reshape(
            self.time_steps_per_day, self.num_grids, len(self.weather_columns)
        )
        
        # 11. 提取坐标
        coords = grid_data[['LON_CENTER', 'LAT_CENTER']].values[:self.num_grids]
        
        # 12. 类型转换与保存
        data_3d = data_3d.astype(np.float32)
        grid_coords = coords.astype(np.float32)
        
        time_strs = unique_times.tolist()
        
        with open(bin_file, 'wb') as f:            
            data_3d.tofile(f)
        
        return time_strs, grid_coords
    
    def save_global_metadata(self, global_time_strs, grid_coords):
        """保存全局元数据"""
        global_metadata = {
            'num_features': len(self.weather_columns),
            'num_grids': self.num_grids,
            'time_steps_per_day': self.time_steps_per_day,
            'feature_names': self.weather_columns,
            'coord_names': ['LON_CENTER', 'LAT_CENTER'],
            'dtype': 'float32',
            'log_transformed_features': self.log_transform_columns,
            # [新增] 单位说明
            'units_note': {
                'temperature': 'Celsius',
                'pressure': '100kPa (Bar) - Converted from hPa',
                'humidity': '0-1 Decimal - Converted from %',
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
    data_dir = '/mnt/drive1/pengpeng/storage/sz_weather/filled_data'
    test_dir = '/mnt/drive1/pengpeng/storage/sz_weather/bin_data'
    
    converter = CSVToBinConverter(data_dir, test_dir)
    converter.process_all_files()
    print("预处理完成！")