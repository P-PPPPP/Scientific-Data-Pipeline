# 🌍 ERA5 逐小时单层数据处理模块 (ERA5-Single-Levels)

本目录提供了针对 ERA5 全球大气再分析数据的全流程处理方案。

## 📜 官方简介 (Official Introduction)

> **ERA5** 是欧洲中期天气预报中心（ECMWF）发布的第五代全球气候与天气再分析数据集。它利用**数据同化（Data Assimilation）**技术，将物理模型与来自全球的观测数据相结合，生成了一套完整且一致的全球大气状态估计。
>
> 相比前代，ERA5 提供了更高的时间分辨率（逐小时）和空间分辨率，涵盖了过去 80 年（1940年至今）的多种大气、陆地和海洋变量。对于气候研究、数值天气预报（NWP）以及物理神经网络的训练具有核心价值。

---

## 📊 数据概览 (Data Specification)

* **来源**：[Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
* **核心变量**：支持包含温度、气压、风场在内的 **19 类** 关键气象变量。
* **时间精度**：1 小时 (Hourly)。
* **存储格式**：
    * **原始阶段**：按天存储的 `.csv` 文件。
    * **深度学习阶段**：高性能二进制 `.bin` 文件，并伴随 `.json` 元数据。

---

## 📂 目录结构 (Directory Structure)

```text
.
├── data_process/                           # 数据获取与转换
│   ├── a_download_data_chunk_days.py       # 基于天数块的下载器
│   └── b_CSVtoBinary_multiprocess.py       # 多进程二进制转换工具
├── pytorch_dataset/                        # 深度学习适配器
│   └── binary_filelist_dataset-4-spatial_interpolation.py  # 空间插值数据集加载器
└── README.md                               # 本文档
```

---

## 🛠 处理流程 (Workflow)

### 1. 数据下载 (Download)
使用 `a_download_data_chunk_days.py`。该脚本支持：
* 指定连续时间段、地理范围及分辨率。
* **断点续传**：自动识别已下载天数，规避重复任务。

### 2. 预处理与二进制转换 (Preprocessing)
运行 `b_CSVtoBinary_multiprocess.py`。
* 该程序会将原始 CSV 转换为适合高效读取的二进制文件。
* 自动导出：`metadata.json` (统计信息), `date_data.json` (日期) 和 `coords_data.npy` (经纬度坐标)。

### 3. 高性能加载 (Loading)
使用内置 `torch Dataset`，利用 **内存映射 (Memory Mapping)** 技术。这对于你正在开发的 **GSF-GAT** 等模型至关重要，因为它能在处理超大规模气象序列时保持极低的 IO 开销。

---

## 🖼 可视化与图结构 Demo (Visualization & Graph Tools)

### 📈 数据可视化预览

*（此处可放置处理后的 2D 场热力图，用于校验空间分布是否正确）*

### 🕸 邻接矩阵生成 (Adjacency Matrix)

*（占位符：针对图神经网络 GSF-GAT 设计，展示基于经纬度或物理相关性构建的节点连接关系）*

### 🎭 掩码生成 (Mask Generation)

*（占位符：用于空间插值或缺失值重构任务的任务掩码示意图）*

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备
```bash
pip install cdsapi pandas numpy torch
# 请确保 ~/.cdsapirc 已正确配置 API Key
```

### 2. 代码示例
你可以通过以下方式直接调用处理好的二进制数据：

```python
from pytorch_dataset.binary_filelist_dataset import SpatialInterpolationDataset

# 初始化数据集（支持内存映射）
dataset = SpatialInterpolationDataset(bin_dir='./data/bin/', metadata_path='./data/metadata.json')
print(f"Total samples: {len(dataset)}")
```

---

## 🤝 后续工作框架 (To-Do List)

* [ ] 补充 `requirements.txt`。
* [ ] 完善 19 个变量的具体预处理逻辑文档。
* [ ] 集成基于图结构的邻接矩阵生成脚本，支持自定义距离阈值。