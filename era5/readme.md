# 🌍 ERA5 Hourly Single-Level Data Processing Module

<p align="center">
  <b>English</b> | <a href="./readme_cn.md">中文说明</a>
</p>

> This directory provides a complete processing pipeline for ERA5 global atmospheric reanalysis data. It is designed as a submodule for climate, weather, or physics-informed neural network (PINN) projects in the main repository.

---

## 📜 Official Introduction

**ERA5** is the fifth generation of global climate and weather reanalysis datasets released by the European Centre for Medium-Range Weather Forecasts (ECMWF). Using **data assimilation** techniques, it combines physical models with observational data from around the world to produce a complete and consistent estimate of the global atmospheric state.

Compared to its predecessors, ERA5 offers higher temporal (hourly) and spatial resolution, covering a wide range of atmospheric, land, and ocean variables from **1940 to the present**. It is of core value for climate research, numerical weather prediction (NWP), and physics-informed neural network training.

---

## 📊 Data Specification

- **Source**: [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
- **Core Variables**: Supports **19 key meteorological variables** (temperature, pressure, wind, etc.)
- **Temporal Resolution**: Hourly
- **Storage Format**:
  - Raw data: Daily `.csv` files
  - Training-ready: High-performance binary `.bin` files + `.json` metadata

---

## 📂 Directory Structure (within this module)

```text
.
├── data_process/
│   ├── a_download_data_chunk_days.py       # Chunk-based downloader
│   ├── b_CSVtoBinary_multiprocess.py       # Multi-process preprocessing + binary conversion
│   └── c_get_elevation.py                  # Retrieve grid point elevation
├── pytorch_dataset/
│   └── binary_filelist_dataset-4-spatial_interpolation.py  # Spatial interpolation dataset loader
└── README.md                               # This document
```

---

## 🔐 API License & Configuration

1. Register and obtain access to [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download).
2. Follow the [CDS API setup guide](https://cds.climate.copernicus.eu/how-to-api) to configure `~/.cdsapirc`.
3. Install `cdsapi`:

```bash
pip install cdsapi
```

### Example request code

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

## 🌟 Recommended Variables (17 in total)

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

### Categorized Description

#### Category 1: Direct Mapping Variables
| Variable | Corresponding Observation | Purpose |
|----------|---------------------------|---------|
| `2m_temperature` | Air temperature | Directly corresponds to AWS observations |
| `2m_dewpoint_temperature` | Relative humidity (to be calculated) | Compute RH and air density with `t2m` |
| `surface_pressure` | Pressure | Requires vertical correction |
| `10m_u/v_component_of_wind` | Wind speed/direction | Compute full wind speed and direction |
| `instantaneous_10m_wind_gust` | Maximum wind gust | Fill extreme wind events (typhoons, convection) |
| `total_precipitation` | Accumulated rainfall | Hourly precipitation (m → mm) |
| `mean_sea_level_pressure` | Pressure (validation) | Eliminate altitude interference |

#### Category 2: Physical Drivers & Auxiliary Inference
| Variable | Target Association | Purpose |
|----------|--------------------|---------|
| `surface_solar_radiation_downwards` | Air temperature, surface temperature | Direct driver of temperature rise |
| `total_cloud_cover` | Visibility, diurnal temperature range | Affects radiative cooling and warming |
| `low_cloud_cover` | Visibility (fog) | Fill missing visibility data |
| `boundary_layer_height` | Visibility, air quality | Affects pollutant dispersion |
| `skin_temperature` | Urban heat island | More sensitive to radiation than 2m temperature |
| `convective_available_potential_energy` | Heavy precipitation, thunderstorm winds | Discriminate convective potential |
| `total_column_water_vapour` | Precipitation, humidity | Total atmospheric water vapor content |

#### Category 3: Geo-static Variables
| Variable | Purpose |
|----------|---------|
| `geopotential` | Compute terrain height (`Height = z / 9.8`) for lapse rate correction |
| `land_sea_mask` | Distinguish land/sea/coastline, explain thermal differences |

---

## 🛠️ Complete Processing Workflow (executed within this submodule)

### 1️⃣ Data Download
Use `a_download_data_chunk_days.py`  
- Supports resuming interrupted downloads  
- Configure time range, area, and resolution directly in the script

```bash
# Execute from the module root directory
nohup python -u ./data_process/a_download_data_chunk_days.py > ./download_task.log &
```

### 2️⃣ Preprocessing + Binary Conversion
Use `b_CSVtoBinary_multiprocess.py`  
- Automatically generates: `metadata.json`, `date_data.json`, `coords_data.npy`, and `.bin` files

```bash
nohup python -u ./data_process/b_CSVtoBinary_multiprocess.py > ./preprocess_task.log &
```

### 3️⃣ Retrieve Grid Elevation (optional)
```bash
python ./data_process/c_get_elevation.py
```

### 4️⃣ PyTorch Dataset Loading
Use `pytorch_dataset/binary_filelist_dataset-4-spatial_interpolation.py`  
- Based on memory mapping  
- Supports spatial interpolation, suitable for very large sequences

---

## 🚀 Quick Start (using this module independently)

```bash
# 1. Enter the submodule directory
cd your_main_repo/era5

# 2. Install dependencies
pip install -r requirements.txt   # if available; otherwise at least: numpy, xarray, cdsapi, torch

# 3. Configure CDS API
# Ensure ~/.cdsapirc contains:
# url: https://cds.climate.copernicus.eu/api
# key: <UID>:<API-Key>

# 4. Download → Preprocess → Train
```