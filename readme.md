  # Scientific-Data-Pipeline

A full-process dataset repository designed for **scientific computing**, **fluid dynamics**, and **spatio-temporal sequence prediction**.

<p align="center">
  <b>English</b> | <a href="./readme_cn.md">中文说明</a>
</p>

> [!WARNING]
> Project Status: This repository is a Work in Progress. The code is primarily intended for academic research and has not been extensively tested in large-scale production environments. There may be bugs in computational logic or implementation. Please perform your own verification and validation when using for research purposes.

-----

## <a id="Navigation">📖 Table of Contents</a>

  - **[Included Datasets Overview](#data-included)**: Quickly understand the background and applicable scenarios of each dataset.
  - **[Dataset Package Downloads](#download-summary)**: Direct access to pre-processed binary data packages.

-----

## <a id="Welcome">🌟 Welcome</a>

When conducting deep learning research in scientific computing (e.g., fluid simulation, weather prediction), obtaining high-quality datasets is often the first hurdle.

Although many public platforms exist, individual researchers frequently face the following pain points:

1.  **High Acquisition Cost**: Downloading, cleaning, and aligning large-scale raw data is extremely time-consuming.
2.  **Difficulty in Reproduction**: Private preprocessing logic in papers makes it hard to benchmark experimental results.
3. **Broken Toolchain**: The conversion process from raw tables to PyTorch `Dataset` is tedious and repetitive.

**Scientific-Data-Pipeline** exists not only to provide data but also to open-source the **complete code flow from data download and preprocessing to Torch Dataset construction**. We pursue research **transparency** and **usability**, aiming to provide reliable foundational support for the intersection of computational mathematics and deep learning.

> [!WARNING]
> **Note**: This project is not built for "leaderboard chasing." If you need standard time-series benchmarking, we recommend [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

-----

## <a id="data-included">📂 Included Datasets</a>

### 🌍 Meteorological Reconstruction & Multi-scale Datasets

#### 1. Shenzhen Automatic Weather Station Data (Urban Scale)
  - **Description**: High-density regional monitoring data from Shenzhen's automatic weather stations. The extremely dense node distribution makes it ideal for high-fidelity spatial interpolation tasks and constructing complex graph neural network (e.g., GAT, GCN) adjacency matrices.
  - **Source**: [Shenzhen Government Data Open Platform](https://opendata.sz.gov.cn/)
  - **Processing & Loader**: [Preprocessing Scripts & Documentation](./sz_weather/readme.md)

#### 2. ERA5-CN (National Scale)
  - **Description**: China regional dataset extracted from global ERA5 reanalysis data. Moderate resolution, ideal for regional spatio-temporal prediction, data assimilation research, and large-scale model pre-training.
  - **Source**: [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
  - **Processing & Loader**: [Preprocessing Scripts & Documentation](./era5/readme.md)

#### 3. ERA5-Global (Global Scale)
  - **Description**: Planetary-scale dataset covering the entire globe. Essential for training global weather prediction foundation models or generating initial/boundary conditions for numerical partial differential equation (PDE) solvers.
  - **Source**: [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
  - **Offline Package**: [Google Drive](https://drive.google.com/file/d/1_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view)
  - **Processing & Loader**: [Preprocessing Scripts & Documentation](./era5/readme.md)

-----

### 🌊 Scientific Computing & Fluid Dynamics Datasets

#### 4. Transonic Wing Flow Multi-Fidelity Dataset
- **Description**: This dataset originates from intermediate results of compressible fluid numerical computations, simulating the physical behavior of a wing in transonic flow fields. Data includes 5 core physical quantities: Density, Mach Number, Pressure, Temperature, Velocity, along with their gradients.
- **Characteristics**: Provides 8 CSV files of varying fidelity levels, covering numerical results from coarse to fine grids. Note that grid positions across different scales are not forcibly aligned, presenting realistic challenges for multi-fidelity learning.
- **Application**: Originally developed for neural network-based high-fidelity data reconstruction and super-resolution simulation, also suitable for spatial interpolation studies of fluid properties.
- **Detailed Config**: [View YAML definition](./fluid_dynamics/naca.yaml)
- **Processing & Loader**: [Fluid Data Preprocessing Documentation](./fluid_dynamics/readme.md)

-----

### 🚦 Spatio-Temporal Forecasting Datasets

#### 5. METR-LA / PEMS-BAY
  - **Description**: Classic traffic speed datasets for Los Angeles and the Bay Area. Commonly used for validating graph dynamics models like Diffusion Convolutional Recurrent Neural Networks (DCRNN).
  - **Reference**: Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting" (ICLR 2018).
  - **Reference Implementation**: [liyaguang/dcrnn](https://github.com/liyaguang/dcrnn)
  - **Offline Package**: [Google Drive - Common ST-Sequence](https://drive.google.com/file/d/1BKZ8Iqfo2x610sUxGEMCjpFYyn_UllTK/view?usp=sharing)
  - **Processing & Loader**: [Data Preprocessing & Dataset Construction](./time_series/readme.md)

#### 6. PEMS-04 / PEMS-08
  - **Description**: California highway traffic flow datasets. Widely used as benchmarks for attention-based spatio-temporal graph convolutional networks (ASTGCN).
  - **Reference**: Guo et al., "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting" (AAAI 2019).
  - **Reference Implementation**: [Davidham3/ASTGCN-2019-mxnet](https://github.com/Davidham3/ASTGCN-2019-mxnet)
  - **Offline Package**: [Google Drive - Common ST-Sequence](https://drive.google.com/file/d/1BKZ8Iqfo2x610sUxGEMCjpFYyn_UllTK/view?usp=sharing)
  - **Processing & Loader**: [Data Preprocessing & Dataset Construction](./time_series/readme.md)

#### 7. TaxiBJ
  - **Description**: Beijing taxi trajectory dataset containing grid-based flow data along with accompanying weather and holiday information, suitable for spatio-temporal prediction tasks with multi-source feature fusion.
  - **Reference**: Zhang et al., "Deep Spatio-Temporal Residual Networks for Citywide Crowd Flow Prediction" (AAAI 2017).
  - **Offline Package**: [Google Drive - Common ST-Sequence](https://drive.google.com/file/d/1BKZ8Iqfo2x610sUxGEMCjpFYyn_UllTK/view?usp=sharing)
  - **Processing & Loader**: [Data Preprocessing & Dataset Construction](./time_series/readme.md)

-----

### 📈 Time-Series Forecasting Datasets

#### 8. ETT (Electricity Transformer Temperature)
  - **Description**: Contains transformer oil temperature and 6 power load features. Available in 15-minute (ETTm) and 1-hour (ETTh) granularities; a benchmark dataset for Long Sequence Time-Series Forecasting (LSTF).
  - **Reference**: Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021 Best Paper).
  - **Source Code**: [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
  - **Offline Package**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **Processing & Loader**: [Data Preprocessing & Dataset Construction](./time_series/readme.md)

#### 9. Electricity / Traffic
  - **Description**: General time-series datasets covering electricity consumption (kWh) and road occupancy (Occupancy) metrics, commonly used for multivariate long-range forecasting tasks.
  - **Reference**: Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" (SIGIR 2018).
  - **Source Code**: [laiguokun/LSTNet](https://github.com/laiguokun/LSTNet)
  - **Offline Package**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **Processing & Loader**: [Data Preprocessing & Dataset Construction](./time_series/readme.md)

#### 10. Weather
  - **Description**: General time-series dataset containing 21 meteorological indicators including temperature, humidity, and air pressure, sourced from the MPI-BGC weather station.
  - **Official Source**: [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
  - **Offline Package**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **Processing & Loader**: [Data Preprocessing & Dataset Construction](./time_series/readme.md)

#### 11. Exchange Rate
  - **Description**: Daily exchange rate monitoring data for 8 countries (Australia, UK, Canada, etc.) from 1990 to 2016.
  - **Reference**: Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" (SIGIR 2018).
  - **Source Code**: [laiguokun/LSTNet](https://github.com/laiguokun/LSTNet)
  - **Offline Package**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **Processing & Loader**: [Data Preprocessing & Dataset Construction](./time_series/readme.md)

#### 12. Illness (ILI)
  - **Description**: Weekly influenza-like illness (ILI) surveillance data from the US CDC, containing the proportion of ILI patients among outpatient visits. A classic task for evaluating long-range modeling capabilities of epidemic forecasting models.
  - **Official Source**: [CDC FluView Portal](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)
  - **Offline Package**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **Processing & Loader**: [Data Preprocessing & Dataset Construction](./time_series/readme.md)

-----

## <a id="download-summary"></a> ⬇️ One-Click Dataset Download Summary

| Dataset Category | Quick Download Link (G-Drive) |
| :--- | :--- |
| **ERA5-Global Preprocessed Binary Package** | [Download](https://drive.google.com/file/d/1_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view) |
| **General Spatio-Temporal Sequence Datasets (ST-Sequence)** | [Download](https://drive.google.com/file/d/1BKZ8Iqfo2x610sUxGEMCjpFYyn_UllTK/view?usp=sharing) |
| **General Time-Series Benchmarks** | [Download](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing) |

-----

## 🤝 Contributions & Feedback

If you encounter bugs while using the code, or wish to contribute new datasets, please feel free to submit an **Issue** or **Pull Request**.