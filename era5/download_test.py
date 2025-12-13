import cdsapi


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

download_filename = f"temp_download.zip"

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": VARIABLES,
        "year": ["2020"],
        "month": ["02"],
        "day": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12",
        "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28"
    ],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "grid": GRID, 
        "format": "netcdf", 
    },
    download_filename
)

print("下载完成:", download_filename)