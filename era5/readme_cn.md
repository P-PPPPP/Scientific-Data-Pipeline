# 🌍 ERA5 逐小时单层数据处理模块

<p align="center">
  <b>中文说明</b> | <a href="./readme.md">English</a>
</p>

> 本目录为 ERA5 全球大气再分析数据的全流程处理子模块，适用于主仓库中的气候、天气或物理信息神经网络（PINN）相关项目。

---

## 📜 官方简介

**ERA5** 是欧洲中期天气预报中心（ECMWF）发布的第五代全球气候与天气再分析数据集。它利用**数据同化（Data Assimilation）** 技术，将物理模型与全球观测数据结合，生成一套完整、一致的全球大气状态估计。

相比前代，ERA5 提供了更高的时间分辨率（逐小时）和空间分辨率，覆盖 **1940 年至今** 的多种大气、陆地和海洋变量。对于气候研究、数值天气预报（NWP）以及物理神经网络的训练，ERA5 具有核心价值。

---

## 📊 数据概览

- **来源**：[Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
- **核心变量**：支持 **19 类** 关键气象变量（温度、气压、风场等）
- **时间精度**：1 小时
- **存储格式**：
  - 原始数据：按天存储的 `.csv`
  - 训练就绪：高性能二进制 `.bin` + `.json` 元数据

---

## 📂 目录结构（本模块内）

```text
.
├── data_process/
│   ├── a_download_data_chunk_days.py       # 按天数块下载
│   ├── b_CSVtoBinary_multiprocess.py       # 多进程预处理 + 二进制转换
│   └── c_get_elevation.py                  # 获取网格点海拔
├── pytorch_dataset/
│   └── binary_filelist_dataset-4-spatial_interpolation.py  # 空间插值数据集加载器
└── README.md                               # 本文档
```

---

## 🔐 API 许可与配置

1. 注册并获取 [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download) 的访问权限。
2. 按照 [CDS API 设置指南](https://cds.climate.copernicus.eu/how-to-api) 配置 `~/.cdsapirc`。
3. 安装 `cdsapi`：

```bash
pip install cdsapi
```

### 示例请求代码

```python
import cdsapi

client = cdsapi.Client()
dataset = 'reanalysis-era5-single-levels'
request = {
    'product_type': ['reanalysis'],
    'variable': ['2m_temperature'],
    'year': ['2024'],
    'month': ['03'],
    'day': ['01'],
    'time': ['13:00'],
    'data_format': 'grib',
}
client.retrieve(dataset, request, 'download.grib')
```

---

## 🌟 推荐下载变量（共 17 类）

```python
ERA5_VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature",
    "2m_temperature", "mean_sea_level_pressure", "surface_pressure",
    "total_precipitation", "instantaneous_10m_wind_gust", "surface_solar_radiation_downwards",
    "low_cloud_cover", "total_cloud_cover", "boundary_layer_height",
    "convective_available_potential_energy", "geopotential", "land_sea_mask",
    "total_column_water_vapour", "skin_temperature"
]
```

### 分类说明

#### 第一类：直接映射变量
| 变量 | 对应观测 | 用途 |
|------|----------|------|
| `2m_temperature` | 气温 | 直接对应 AWS 观测 |
| `2m_dewpoint_temperature` | 相对湿度（需计算） | 结合 `t2m` 计算 RH 与空气密度 |
| `surface_pressure` | 气压 | 需垂直订正 |
| `10m_u/v_component_of_wind` | 风速/风向 | 计算全风速与风向 |
| `instantaneous_10m_wind_gust` | 极大风速 | 台风/强对流填补 |
| `total_precipitation` | 累计降雨量 | 小时降水量（米→毫米） |
| `mean_sea_level_pressure` | 气压（校验） | 消除海拔干扰 |

#### 第二类：物理驱动与辅助推断
| 变量 | 关联目标 | 用途 |
|------|----------|------|
| `surface_solar_radiation_downwards` | 气温、地表温 | 温度上升的直接驱动力 |
| `total_cloud_cover` | 能见度、气温日较差 | 影响辐射降温与升温 |
| `low_cloud_cover` | 能见度（大雾） | 填补能见度缺失 |
| `boundary_layer_height` | 能见度、空气质量 | 影响污染物扩散 |
| `skin_temperature` | 城市热岛 | 辐射响应更剧烈 |
| `convective_available_potential_energy` | 强降水、雷暴大风 | 对流潜力判别 |
| `total_column_water_vapour` | 降水、湿度 | 水汽总含量 |

#### 第三类：地理静态变量
| 变量 | 用途 |
|------|------|
| `geopotential` | 计算地形高度（`Height = z / 9.8`），用于垂直递减率订正 |
| `land_sea_mask` | 区分陆地/海洋/海岸线，解释热力差异 |

---

## 🛠️ 完整处理流程（子模块内执行）

### 1️⃣ 数据下载
使用 `a_download_data_chunk_days.py`  
- 支持断点续传  
- 在脚本内直接配置时间段、范围、分辨率

```bash
# 在模块根目录下执行
nohup python -u ./data_process/a_download_data_chunk_days.py > ./download_task.log &
```

### 2️⃣ 预处理 + 二进制转换
使用 `b_CSVtoBinary_multiprocess.py`  
- 自动生成：`metadata.json`、`date_data.json`、`coords_data.npy` 及 `.bin` 文件

```bash
nohup python -u ./data_process/b_CSVtoBinary_multiprocess.py > ./preprocess_task.log &
```

### 3️⃣ 获取网格海拔（可选）
```bash
python ./data_process/c_get_elevation.py
```

### 4️⃣ PyTorch 数据集加载
使用 `pytorch_dataset/binary_filelist_dataset-4-spatial_interpolation.py`  
- 基于内存映射（Memory Mapping）  
- 支持空间插值，适合超大规模序列

---

## 🚀 快速上手（独立使用本模块）

```bash
# 1. 进入子模块目录
cd your_main_repo/era5

# 2. 安装依赖
pip install -r requirements.txt   # 如果存在，否则至少需要：numpy, xarray, cdsapi, torch

# 3. 配置 CDS API
# 确保 ~/.cdsapirc 包含：
# url: https://cds.climate.copernicus.eu/api
# key: <UID>:<API-Key>

# 4. 下载 → 预处理 → 训练
```