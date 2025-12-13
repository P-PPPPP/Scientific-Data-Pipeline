import cdsapi
import xarray as xr
import pandas as pd
import os
import datetime
import zipfile
import glob
import shutil

# ================= 配置区域 =================
# 数据保存的文件夹路径
OUTPUT_DIR = "./era5_daily_data"

# 变量列表 (保持不变)
VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature",
    "2m_temperature", "mean_sea_level_pressure", "surface_pressure",
    "total_precipitation", "instantaneous_10m_wind_gust", "surface_solar_radiation_downwards",
    "low_cloud_cover", "total_cloud_cover", "boundary_layer_height",
    "convective_available_potential_energy", "geopotential", "land_sea_mask",
    "total_column_water_vapour", "skin_temperature"
]

AREA = [54, 73, 3, 135]
GRID = [1, 1]  # 分辨率设置为 1 度
c = cdsapi.Client()

def download_and_process_day(target_date):
    year = target_date.strftime("%Y")
    month = target_date.strftime("%m")
    day = target_date.strftime("%d")
    date_str = f"{year}{month}{day}"
    
    final_csv_path = os.path.join(OUTPUT_DIR, f"{date_str}.csv")

    if os.path.exists(final_csv_path):
        print(f"⏭️  {date_str} 的数据已存在，跳过。")
        return

    print(f"🔄 正在处理: {year}-{month}-{day}...")
    
    download_filename = f"temp_download_{date_str}.zip"
    extract_folder = f"temp_extract_{date_str}"

    try:
        # 1. 下载 (保持不变)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": VARIABLES,
                "year": year,
                "month": month,
                "day": day,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": AREA,
                "grid": GRID, 
                "format": "netcdf", 
            },
            download_filename
        )

        # 2. 解压 (保持不变)
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)
        
        if zipfile.is_zipfile(download_filename):
            with zipfile.ZipFile(download_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
        else:
            shutil.move(download_filename, os.path.join(extract_folder, "data.nc"))

        # 3. 读取 NC 文件
        nc_files = glob.glob(os.path.join(extract_folder, "*.nc"))
        if not nc_files:
            raise Exception("没有找到 .nc 文件")

        dataframes = []
        for nc_file in nc_files:
            ds = xr.open_dataset(nc_file, engine="netcdf4")
            df_temp = ds.to_dataframe().reset_index()
            ds.close()

            # === 列名清洗逻辑 ===
            if 'valid_time' in df_temp.columns:
                df_temp.rename(columns={'valid_time': 'DDATETIME'}, inplace=True)
            elif 'time' in df_temp.columns:
                df_temp.rename(columns={'time': 'DDATETIME'}, inplace=True)
            else:
                raise KeyError(f"文件 {os.path.basename(nc_file)} 找不到时间列")

            if 'longitude' in df_temp.columns:
                df_temp.rename(columns={'longitude': 'LON_CENTER'}, inplace=True)
            if 'latitude' in df_temp.columns:
                df_temp.rename(columns={'latitude': 'LAT_CENTER'}, inplace=True)

            # === 【新增】 强力清洗多余列 ===
            cols_to_keep = ['DDATETIME', 'LON_CENTER', 'LAT_CENTER'] 
            
            # 需要剔除的黑名单列
            drop_list = ['number', 'step', 'surface', 'heightAboveGround', 
                         'entireAtmosphere', 'expver'] # <--- 在这里加入了 expver

            for col in df_temp.columns:
                if col in cols_to_keep: continue
                if col in drop_list: continue
                cols_to_keep.append(col)
            
            df_temp = df_temp[cols_to_keep]

            # === 【关键】 去重 ===
            # 去除因为 expver 不同但时间地点相同导致的重复行
            # 并处理可能的 NaN 值（通常 expver=1 有值则 5 为空，反之亦然）
            # groupby().first() 是一个简单有效的合并策略
            df_temp = df_temp.groupby(['DDATETIME', 'LON_CENTER', 'LAT_CENTER'], as_index=False).first()

            dataframes.append(df_temp)

        # 4. 合并数据
        full_df = dataframes[0]
        if len(dataframes) > 1:
            for i in range(1, len(dataframes)):
                full_df = pd.merge(
                    full_df, 
                    dataframes[i], 
                    on=["DDATETIME", "LON_CENTER", "LAT_CENTER"], 
                    how="outer"
                )

        # 5. 格式化并保存
        full_df['DDATETIME'] = pd.to_datetime(full_df['DDATETIME']).dt.strftime('%Y-%m-%d %H:%M:%S')
        full_df.to_csv(final_csv_path, index=False)
        print(f"  ✅ 已保存: {final_csv_path} ({len(full_df)} 行)")

    except Exception as e:
        print(f"  ❌ 处理 {date_str} 出错: {e}")
        if os.path.exists(final_csv_path):
            os.remove(final_csv_path)
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(download_filename):
            os.remove(download_filename)
        if os.path.exists(extract_folder):
            shutil.rmtree(extract_folder)

def main():
    # 1. 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建目录: {OUTPUT_DIR}")

    # 2. 设置时间范围: 2020-01-01 到 2025-12-31
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2025, 1, 1)
    
    # 获取今天日期（避免请求未来数据导致报错）
    today = datetime.date.today()
    # ERA5 正式数据有约5天延迟，ERA5T 也要几天延迟，这里做一个简单的保护
    # 如果你想请求到昨天的数据，可以改为 today - datetime.timedelta(days=1)
    max_allowed_date = today 

    delta = datetime.timedelta(days=1)
    current_date = start_date

    while current_date <= end_date:
        # 如果当前循环日期超过了今天，提前终止
        if current_date > max_allowed_date:
            print(f"⚠️  日期 {current_date} 在未来，停止处理。")
            break
            
        download_and_process_day(current_date)
        current_date += delta

    print(f"\n全部任务完成！文件保存在 {OUTPUT_DIR}")

if __name__ == "__main__":
    pid = os.getpid()
    print(f"当前进程 ID: {pid}")
    main()