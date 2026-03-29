# Scientific-Data-Pipeline

A full-lifecycle data pipeline designed for **Scientific Computing**, **Fluid Dynamics**, and **Spatio-Temporal Forecasting**.

<p align="center">
  <b>English</b> | <a href="./readme_cn.md">中文说明</a>
</p>

> [!WARNING]
> **Work in Progress**: This repository is under active development. Some scripts may contain bugs or unoptimized logic. Please use with caution and double-check results for critical research tasks.

-----

## 📖 Navigation

  - **[Dataset Introduction](https://www.google.com/search?q=%23datasets-included)**: Overview of backgrounds and application scenarios.
  - **[Dataset Statistics](https://www.google.com/search?q=./data_stats.md)**: Detailed metrics including sample counts and feature dimensions.
  - **[Data Download](https://www.google.com/search?q=%23data-download)**: Direct access to pre-processed binary data packages.

-----

## 🌟 Mission Statement (Welcome)

In research involving deep learning for scientific computing (e.g., fluid simulation, meteorological prediction), obtaining high-quality datasets is often the first and most significant hurdle.

While many public platforms exist, individual researchers often face:

1.  **High Acquisition Costs**: Downloading, cleaning, and aligning large-scale raw data is extremely time-consuming.
2.  **Reproducibility Issues**: Private preprocessing logic in papers makes it difficult to benchmark experimental results.
3.  **Toolchain Gaps**: The transition from raw tabular files to PyTorch `Dataset` classes is often tedious and repetitive.

**Scientific-Data-Pipeline** provides more than just data; it discloses the **entire workflow**—from downloading and preprocessing to building Torch Datasets. Our goal is to ensure research is **reproducible, reliable, and accessible**, supporting the intersection of computational mathematics and deep learning.

> [!WARNING]
> **Note**: This repository is not intended for "leaderboard chasing." If you require standard time-series benchmark evaluations, we recommend [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

-----

## 📂 Datasets Included

### 🌍 Meteorological & Multi-Scale Datasets

#### 1\. Shenzhen Automatic Weather Station Data (City-Scale)

  - **Description**: High-density regional monitoring data from automatic weather stations in Shenzhen. Its dense node distribution is ideal for high-fidelity spatial interpolation and constructing adjacency matrices for graph-based models like GAT or GCN.
  - **Source**: Shenzhen Government Open Data Platform.
  - **Processing Note**: Raw data is tabular; it is recommended to convert these into `.npy` or `.bin` formats for efficient deep learning training.
  - **download**: [Pending, link to download]
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

#### 2\. ERA5-CN (National-Scale)

  - **Description**: Extracted from the global ERA5 reanalysis specifically for the China region. This provides a robust resolution suitable for regional spatio-temporal forecasting and data assimilation research.
  - **Source**: Copernicus Climate Data Store (CDS).
  - **download**: [Pending, link to download]
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

#### 3\. ERA5-Global (Planetary-Scale)

  - **Description**: A global snapshot covering the entire Earth manifold. Essential for training large-scale weather prediction models or generating boundary conditions for numerical PDE solvers.
  - **Source**: Copernicus Climate Data Store (CDS).
  - **download**: [https://drive.google.com/file/d/1\_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view](https://drive.google.com/file/d/1_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view)
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

-----

### 🚦 Spatio-Temporal Forecasting Datasets

#### 4\. METR-LA / PEMS-BAY

  - **Description**: Classic traffic speed datasets recorded by sensors in Los Angeles and the Bay Area. Widely used for validating graph dynamics models like DCRNN.
  - **download**: [Pending, link to download]
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

#### 5\. PEMS-04 / PEMS-08

  - **Description**: Traffic flow data from California highways (San Francisco and San Bernardino). Standard benchmarks for attention-based spatio-temporal graph networks (ASTGCN).
  - **download**: [Pending, link to download]
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

#### 6\. TaxiBJ

  - **Description**: Taxicab trajectory data in Beijing. Includes grid-based inflow/outflow alongside meteorological and holiday metadata.
  - **download**: [Pending, link to download]
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

-----

### 📈 Time-Series Forecasting Datasets

#### 7\. ETT (Electricity Transformer Temperature)

  - **Description**: Includes oil temperature and six power load features from electricity transformers. Available in 15-minute (ETTm) and 1-hour (ETTh) granularities.
  - **download**: [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

#### 8\. Electricity / Traffic / Weather

  - **Description**: Common datasets covering power consumption, road occupancy, and 21 meteorological indicators (temperature, humidity, etc.).
  - **download**: [https://drive.google.com/file/d/1rHJYc8cgNFPPvWLRpwynGj2xohqcc2R7/view?usp=sharing](https://drive.google.com/file/d/1rHJYc8cgNFPPvWLRpwynGj2xohqcc2R7/view?usp=sharing)
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

#### 9\. Exchange Rate / Illness (ILI)

  - **Description**: Tracks daily exchange rates of eight countries and weekly influenza-like illness ratios from the CDC.
  - **download**: [Pending, link to download]
  - **Process Code**: [Pending, link to readme in preprocessing directory]
  - **Reference Dataset Class**: [Pending, link to directory]

-----

## ⬇️ Data Download Summary

Direct links to pre-processed data packages for quick experimentation:

| Category | Download Link |
| :--- | :--- |
| **ERA5-Global Binary Package** | [Google Drive](https://drive.google.com/file/d/1_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view) |
| **Common Time-Series Datasets** | [Google Drive](https://drive.google.com/file/d/1rHJYc8cgNFPPvWLRpwynGj2xohqcc2R7/view?usp=sharing) |

-----

## 🤝 Contribution

If you encounter bugs or wish to contribute new datasets/preprocessing scripts, please feel free to open an **Issue** or submit a **Pull Request**.