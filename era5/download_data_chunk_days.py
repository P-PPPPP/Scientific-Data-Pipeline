import cdsapi
import xarray as xr
import pandas as pd
import os
import datetime
import zipfile
import glob
import shutil
from collections import defaultdict
import traceback

class ERA5Downloader:
    def __init__(self, output_dir, variables, area_grid, chunk_days=5, area=None):
        """
        初始化下载器
        :param output_dir: 数据保存的根目录
        :param variables: 需要下载的变量列表
        :param area_grid: 分辨率 [lat_step, lon_step]
        :param chunk_days: 每次请求打包的天数
        """
        self.output_dir = output_dir
        self.variables = variables
        self.area = area
        self.grid = area_grid
        self.chunk_days = chunk_days
        
        # 初始化 CDS API 客户端
        self.client = cdsapi.Client()
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 创建输出目录: {self.output_dir}")

    def _get_csv_path(self, date_obj):
        """生成最终 CSV 文件路径"""
        date_str = date_obj.strftime("%Y%m%d")
        return os.path.join(self.output_dir, f"{date_str}.csv")

    def _check_missing_dates(self, start_date, end_date):
        """检查哪些日期的数据还未下载"""
        all_dates = []
        curr = start_date
        while curr <= end_date:
            all_dates.append(curr)
            curr += datetime.timedelta(days=1)
        
        missing = [d for d in all_dates if not os.path.exists(self._get_csv_path(d))]
        return missing

    def _download_cds_chunk(self, year, month, days, zip_filename):
        """执行 CDS API 请求"""
        request_dict = {
                "product_type": "reanalysis",
                "variable": self.variables,
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": self.area,
                "grid": self.grid,
                "format": "netcdf",
            }
        
        if self.area is None:
            request_dict.pop('area')

        self.client.retrieve(
            "reanalysis-era5-single-levels",
            request_dict,
            zip_filename
        )

    def _clean_dataframe(self, df):
        """数据清洗逻辑：重命名、删除多余列、处理 expver"""
        # 1. 统一时间列名
        if 'valid_time' in df.columns:
            df.rename(columns={'valid_time': 'DDATETIME'}, inplace=True)
        elif 'time' in df.columns:
            df.rename(columns={'time': 'DDATETIME'}, inplace=True)
        else:
            raise KeyError("❌ 数据中找不到时间列 (time 或 valid_time)")

        # 2. 统一经纬度列名
        if 'longitude' in df.columns:
            df.rename(columns={'longitude': 'LON_CENTER'}, inplace=True)
        if 'latitude' in df.columns:
            df.rename(columns={'latitude': 'LAT_CENTER'}, inplace=True)

        # 3. 筛选保留列
        cols_to_keep = ['DDATETIME', 'LON_CENTER', 'LAT_CENTER'] 
        drop_list = ['number', 'step', 'surface', 'heightAboveGround', 'entireAtmosphere', 'expver']
        
        for col in df.columns:
            if col not in cols_to_keep and col not in drop_list:
                cols_to_keep.append(col)
        
        df = df[cols_to_keep]

        # 4. 去重 (处理 ERA5 的 expver 混合数据问题)
        # groupby first 会取非空的第一条，通常能合并 expver=1 和 expver=5
        df = df.groupby(['DDATETIME', 'LON_CENTER', 'LAT_CENTER'], as_index=False).first()
        
        # 5. 确保时间类型
        df['DDATETIME'] = pd.to_datetime(df['DDATETIME'])
        
        return df

    def _save_daily_csv(self, df):
        """将清洗后的大 DataFrame 按天拆分并保存"""
        df['date_key'] = df['DDATETIME'].dt.date
        grouped = df.groupby('date_key')

        for date_key, group_df in grouped:
            date_str = date_key.strftime("%Y%m%d")
            final_csv_path = os.path.join(self.output_dir, f"{date_str}.csv")
            
            # 格式化时间并保存
            save_df = group_df.drop(columns=['date_key']).copy()
            save_df['DDATETIME'] = save_df['DDATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            save_df.to_csv(final_csv_path, index=False)

    def process_chunk(self, start_date, end_date):
        """核心处理逻辑：处理一个时间块 (下载 -> 清洗 -> 保存)"""
        # 1. 检查缺失日期
        missing_dates = self._check_missing_dates(start_date, end_date)
        if not missing_dates:
            print(f"⏭️  {start_date} 至 {end_date} 全部存在，跳过。")
            return

        print(f"🔄 处理时间段: {start_date} 至 {end_date} (需下载 {len(missing_dates)} 天)...")

        # 2. 准备临时文件夹
        chunk_id = start_date.strftime("%Y%m%d")
        extract_folder = f"temp_extract_chunk_{chunk_id}"
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        # 3. 构建请求映射 (Year, Month) -> [Days]
        requests_map = defaultdict(list)
        for d in missing_dates:
            requests_map[(d.year, d.month)].append(d.strftime("%d"))

        try:
            # 4. 循环下载
            for (year, month), days in requests_map.items():
                print(f"   ⬇️  正在请求 CDS: {year}-{month:02d}, 天数: {len(days)} 天")
                zip_filename = f"temp_download_{chunk_id}_{year}{month:02d}.zip"
                
                self._download_cds_chunk(year, month, days, zip_filename)

                # 解压/移动文件
                if zipfile.is_zipfile(zip_filename):
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder)
                else:
                    # 兼容 API 直接返回 nc 的情况
                    shutil.move(zip_filename, os.path.join(extract_folder, f"data_{year}{month:02d}.nc"))
                
                if os.path.exists(zip_filename):
                    os.remove(zip_filename)

            # 5. 合并 NC 文件
            nc_files = glob.glob(os.path.join(extract_folder, "*.nc"))
            if not nc_files:
                raise FileNotFoundError("下载成功但未在解压目录找到 .nc 文件")

            print(f"   ⚙️  正在解析 {len(nc_files)} 个 NC 文件...")
            
            # 使用 xarray 读取
            with xr.open_mfdataset(nc_files, engine="netcdf4", combine='by_coords', compat='override') as ds:
                df_temp = ds.to_dataframe().reset_index()

            # 6. 清洗数据
            df_cleaned = self._clean_dataframe(df_temp)

            # 7. 保存结果
            print(f"   💾 正在拆分并保存 CSV...")
            self._save_daily_csv(df_cleaned)
            
            print(f"   ✅ 时间段 {start_date} 至 {end_date} 完成。")

        except Exception as e:
            print(f"   ❌ 出错: {e}")
            traceback.print_exc()
        finally:
            # 清理临时目录
            if os.path.exists(extract_folder):
                try:
                    shutil.rmtree(extract_folder)
                except OSError:
                    pass

    def run(self, start_date_str, end_date_str):
        """主入口：遍历整个日期范围"""
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        today = datetime.date.today()
        current_date = start_date

        while current_date <= end_date:
            if current_date > today:
                print(f"⚠️  日期 {current_date} 在未来，停止处理。")
                break
            
            # 计算当前 Chunk 结束时间
            chunk_end = current_date + datetime.timedelta(days=self.chunk_days - 1)
            
            # 边界修正
            if chunk_end > end_date:
                chunk_end = end_date
            if chunk_end > today:
                chunk_end = today

            self.process_chunk(current_date, chunk_end)
            
            current_date = chunk_end + datetime.timedelta(days=1)
        
        print(f"\n🎉 全部任务完成！数据保存在: {self.output_dir}")


# ================= 配置与执行区域 =================
if __name__ == "__main__":
    # 打印进程 ID 方便监控
    print(f"当前进程 ID: {os.getpid()}")

    # 1. 定义变量列表
    ERA5_VARIABLES = [
        "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature",
        "2m_temperature", "mean_sea_level_pressure", "surface_pressure",
        "total_precipitation", "instantaneous_10m_wind_gust", "surface_solar_radiation_downwards",
        "low_cloud_cover", "total_cloud_cover", "boundary_layer_height",
        "convective_available_potential_energy", "geopotential", "land_sea_mask",
        "total_column_water_vapour", "skin_temperature"
    ]

    # 2. 实例化下载器类 (在这里修改主要参数)
    downloader = ERA5Downloader(
        output_dir="./era5_daily_data_global_5*5",  # 保存路径
        variables=ERA5_VARIABLES,               # 变量列表
        area_grid=[5, 5],                       # 分辨率
        # area=[54, 73, 3, 135],
        chunk_days=5                            # 每次请求天数
    )

    # 3. 运行下载任务 (在这里修改时间范围)
    downloader.run(
        start_date_str="2020-01-01",
        end_date_str="2025-12-10"
    )