import numpy as np
import pandas as pd
from pathlib import Path
import json

class CSVToBinConverter:
    def __init__(self, data_dir, target_dir):
        self.data_dir = Path(data_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 原始读取列 (从CSV中读取这些列)
        # 这些是从 ERA5 原始数据中提取的基础变量
        self.keep_columns = [
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
            'blh',              # 边界层高度：决定污染物垂直扩散，关联能见度 (后续转 km)
            'tcwv',             # 整层水汽 (kg/m2 或 mm)：数值范围约 1-70。若全部凝结对应的水深。暴雨发生的绝对水汽条件。
            
            # --- 地表热力与辐射 (Surface Thermal & Radiation) ---
            'skt',              # 肤温 (Skin Temp)：对辐射响应快，用于监测城市热岛效应 (后续转 °C)
            'ssrd',             # 地表短波辐射：气温上升的直接驱动力 (J/m2 -> W/m2 -> Log)
            
            # --- 降水 (Precipitation) ---
            'tp',               # 总降水量：直接对应 AWS 雨量计 (m -> mm -> Log)
            
            # --- 静态地理信息 (Geo-Static) ---
            'z',                # 位势：地势高度 (z/9.8)，用于气温/气压的垂直递减率订正 (后续转 km)
            'lsm'               # 海陆掩码：区分深圳滨海特征 (陆地/海洋热力差异)
        ]
        
        # 2. 最终输出到 .bin 文件的特征列表
        # 模型输入通道顺序 (Channel Order)
        self.final_features = [
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
        
        # 3. 需要进行 Log1p 变换的长尾分布变量
        # 针对数值跨度极大、呈指数分布的变量
        self.log_transform_columns = ['ssrd', 'tp', 'cape', 'i10fg']
        
        self.num_grids = 3276
        self.time_steps_per_day = 24
        
    def calculate_derived_features(self, df):
        """计算物理衍生变量：RH 和 WS"""
        # --- A. 计算相对湿度 (RH) ---
        # Magnus 公式。注意 ERA5 的 t2m/d2m 原始单位通常是 Kelvin
        # 我们在这里先做计算，后续再统一将列转换为摄氏度
        t_c = df['t2m'] - 273.15
        td_c = df['d2m'] - 273.15
        
        # 饱和水汽压 (hPa)
        es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
        # 实际水汽压 (hPa)
        e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
        
        # [修改点 1]：计算 RH，不乘以 100，直接保留小数形式 (0.0 - 1.0)
        df['rh'] = (e / es)
        
        # 物理截断：RH 必须在 0-1 之间
        df['rh'] = df['rh'].clip(0, 1)

        # --- B. 计算合成风速 (WS) ---
        df['ws'] = np.sqrt(df['u10']**2 + df['v10']**2)
        
        return df

    def process_units_and_log(self, df):
        """物理单位修正 和 Log1p 变换"""
        
        # 1. 温度转换: Kelvin -> Celsius
        # 范围约 -30 到 40
        temp_cols = ['t2m', 'd2m', 'skt']
        for col in temp_cols:
            if col in df.columns:
                df[col] = df[col] - 273.15

        # 2. 压力转换: Pascal -> 100 kPa (Bar)
        # 范围约 0.5 到 1.1
        pressure_cols = ['sp', 'msl']
        for col in pressure_cols:
            if col in df.columns:
                df[col] = df[col] / 100000.0

        # [新增] 3. 高度类变量转换: -> Kilometers (km)
        # 将 z 和 blh 统一转换为千米，使其数值保持在 0-10 左右的小数范围
        
        # z: Geopotential (m^2/s^2) -> Geopotential Height (km)
        if 'z' in df.columns:
            # 除以重力加速度 g 和 1000
            df['z'] = df['z'] / 9806.65
        
        # blh: Boundary Layer Height (m) -> (km)
        if 'blh' in df.columns:
            df['blh'] = df['blh'] / 1000.0

        # 4. 百分比变量确保为小数 (0-1)
        cloud_cols = ['lcc', 'tcc']
        for col in cloud_cols:
            if col in df.columns:
                df[col] = df[col].clip(0, 1)

        # 5. 降水 TP: m -> mm
        # Log处理前先转为毫米，数值范围好一些
        if 'tp' in df.columns:
            df['tp'] = df['tp'] * 1000
            df['tp'] = df['tp'].clip(lower=0)

        # 6. 辐射 SSRD: J/m^2 -> W/m^2
        if 'ssrd' in df.columns:
             df['ssrd'] = df['ssrd'] / 3600.0
             df['ssrd'] = df['ssrd'].clip(lower=0)

        # 7. Log1p 变换
        # 针对长尾分布变量 (ssrd, tp, cape, i10fg)
        for col in self.log_transform_columns:
            if col in df.columns:
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
            
            # 处理单个文件
            time_strs, grid_coords = self.process_single_file(csv_file, bin_file)
            global_time_strs[csv_file.stem] = time_strs
            
            if first_grid_coords is None:
                first_grid_coords = grid_coords
        
        # 保存全局元数据
        self.save_global_metadata(global_time_strs, first_grid_coords)
        
    def process_single_file(self, csv_file, bin_file):
        """处理单个CSV文件"""
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 1. 过滤掉不需要的杂列
        existing_cols = [c for c in self.keep_columns if c in df.columns]
        df = df[existing_cols]
        
        # 2. 特征工程 (计算 RH, WS)
        df = self.calculate_derived_features(df)
        
        # 3. 物理单位与 Log 变换 (包含温度转摄氏度、气压转Bar、Log变换)
        df = self.process_units_and_log(df)
        
        # 4. 准备 Grid Merge 数据
        grid_data = df.copy()
        
        expected_len = self.time_steps_per_day * self.num_grids
        if len(grid_data) != expected_len:
             if len(grid_data) > expected_len:
                 grid_data = grid_data.iloc[-expected_len:]
             else:
                 raise ValueError(f"数据长度不足: {len(grid_data)} < {expected_len}")
        
        # 生成/合并 Grid ID
        unique_grids_coords = grid_data[['LON_CENTER', 'LAT_CENTER']].drop_duplicates()
        unique_grids_coords = unique_grids_coords.sort_values(by=['LAT_CENTER', 'LON_CENTER']).reset_index(drop=True)
        unique_grids_coords['GRIDID'] = unique_grids_coords.index

        grid_data = grid_data.merge(unique_grids_coords, on=['LON_CENTER', 'LAT_CENTER'], how='left')

        # 按时间步和网格ID排序
        grid_data = grid_data.sort_values(['DDATETIME', 'GRIDID'])
        
        # 5. 提取最终特征矩阵
        for col in self.final_features:
            if col not in grid_data.columns:
                print(f"Warning: Feature {col} missing in {csv_file.name}, filling with 0.")
                grid_data[col] = 0.0
                
        # 提取值并 Reshape
        weather_features = grid_data[self.final_features].values
        
        data_3d = weather_features.reshape(
            self.time_steps_per_day, self.num_grids, len(self.final_features)
        )
        
        # 提取坐标信息
        coords = grid_data[['LON_CENTER', 'LAT_CENTER']].values
        coords_2d = coords.reshape(self.time_steps_per_day, self.num_grids, 2)
        grid_coords = coords_2d[0] 
        
        # 转换为float32
        data_3d = data_3d.astype(np.float32)
        grid_coords = grid_coords.astype(np.float32)
        
        # 提取时间字符串
        unique_times = grid_data['DDATETIME'].unique()
        time_strs = unique_times.tolist()
        
        # 写入二进制文件
        with open(bin_file, 'wb') as f:            
            data_3d.tofile(f)
        
        return time_strs, grid_coords
    
    def save_global_metadata(self, global_time_strs, grid_coords):
        """保存全局元数据"""
        global_metadata = {
            'num_features': len(self.final_features),
            'num_grids': self.num_grids,
            'time_steps_per_day': self.time_steps_per_day,
            'feature_names': self.final_features,
            'coord_names': ['LON_CENTER', 'LAT_CENTER'],
            'dtype': 'float32',
            'log_transformed_features': self.log_transform_columns,
            # [更新] 记录单位说明
            'units_note': {
                'temperature': 'Celsius',
                'pressure': '100kPa (Bar)',
                'height_z_blh': 'Kilometers (km)',  # 标记 z 和 blh 变成了 km
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
    data_dir = '/mnt/drive1/pengpeng/storage/era5/era5_daily_data'
    target_dir = '/mnt/drive1/pengpeng/storage/era5/era5_bin_data'
    
    converter = CSVToBinConverter(data_dir, target_dir)
    converter.process_all_files()
    print("预处理完成！")