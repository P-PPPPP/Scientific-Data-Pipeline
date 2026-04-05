# Scientific-Data-Pipeline

一个专为**科学计算**、**流体动力学**与**时空序列预测**设计的全流程数据集仓库。

<p align="center">
  <b>中文说明</b> | <a href="./readme.md">English</a>
</p>

> [!WARNING]
> 项目状态：本仓库正处于开发阶段（Work in Progress）。代码主要用于学术研究，尚未经过大规模生产环境测试，可能存在计算逻辑或实现上的 Bug。科研使用时请务必进行校对与验证。

-----

## <a id="Navigation">📖 目录导航</a>

  - **[已收录数据集简介](#data-included)**：快速了解各数据集背景及适用场景。
  - **[数据打包下载](#download-summary)**：直接获取处理好的二进制数据包。

-----

## <a id="Welcome">🌟 Welcome</a>

在开展科学计算（如流体模拟、气象预测）相关的深度学习研究时，获取高质量的数据集往往是第一道门槛。

虽然公开平台很多，但个人研究者常面临以下痛点：

1.  **获取成本高**：大规模原始数据的下载、清洗与对齐极其耗时。
2.  **复现困难**：论文中私有的预处理逻辑导致实验结果难以对标。
3.  **工具链断裂**：从原始表格到 PyTorch `Dataset` 的转换过程枯燥且重复。

**Scientific-Data-Pipeline** 的存在不仅是为了提供数据，更是为了公开**从数据下载、预处理到构建 Torch Dataset 的完整代码流**。我们追求的是研究的**透明度**与**可用性**，旨在为计算数学与深度学习的交叉领域提供可靠的基础支撑。

> [!WARNING]
> **注**：本项目并非为了“刷榜”而建。如果你需要标准的时间序列 Benchmark 评测，推荐前往 [Time-Series-Library](https://github.com/thuml/Time-Series-Library)。

-----

## <a id="data-included">📂 已收录数据集</a>

### 🌍 气象重构与多尺度数据集 (Meteorological Datasets)

#### 1. Shenzhen Automatic Weather Station Data (城市尺度)
  - **描述**：来自深圳市自动气象站的高密度区域监测数据。其节点分布极密，非常适合高保真空间插值任务以及构建复杂的图神经网络（如 GAT, GCN）邻接矩阵。
  - **数据来源**：[深圳政府数据开放平台](https://opendata.sz.gov.cn/)
  - **处理流程与加载器**: [预处理脚本与说明](./sz_weather/readme.md)

#### 2. ERA5-CN (国家尺度)
  - **描述**：从全球 ERA5 再分析资料中提取的中国区域数据集。分辨率适中，是开展区域性时空预测、数据同化研究以及大规模模型预训练的理想选择。
  - **数据来源**：[Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
  - **处理流程与加载器**: [预处理脚本与说明](./era5/readme.md)

#### 3. ERA5-Global (全球尺度)
  - **描述**：覆盖全球的行星级数据集。对于训练全球天气预测大模型或为数值偏微分方程（PDE）求解器生成初始/边界条件至关重要。
  - **数据来源**：[Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
  - **离线包下载**: [Google Drive](https://drive.google.com/file/d/1_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view)
  - **处理流程与加载器**: [预处理脚本与说明](./era5/readme.md)

-----

### 🌊 科学计算与流体动力学数据集 (Scientific Computing & Fluid Dynamics)

#### 4. Transonic Wing Flow Multi-Fidelity Dataset (跨声速机翼流场多置信度数据集)
- **描述**：该数据集来源于可压缩流体数值计算的中间结果，模拟了机翼在临界音速流场中的物理表现。数据包含 密度 (Density)、马赫数 (Mach Number)、压力 (Pressure)、温度 (Temperature) 和 速度 (Velocity) 5 项核心物理量及其梯度。
- **数据特性**：提供 8 个不同精细度的 CSV 文件，涵盖了从粗糙网格到精细网格的数值计算结果。注意：不同尺度间的网格位置未进行强制对齐，这为多置信度学习（Multi-fidelity Learning）提供了真实的挑战场景。
- **应用场景**：最初开发用于神经网络的高保真数据重建（Data Reconstruction）与超分辨率模拟，亦可用于流体特性的空间插值研究。
- **详细配置**：[查看 yaml 定义](./fluid_dynamics/naca.yaml)
- **处理流程与加载器**: [流体数据预处理说明](./fluid_dynamics/readme.md)

-----

### 🚦 时空序列预测数据集 (Spatio-Temporal Forecasting)

#### 5. METR-LA / PEMS-BAY
  - **描述**：经典的交通速度数据集，分别包含洛杉矶和湾区的传感器监测数据。常用于扩散卷积循环神经网络（DCRNN）等图动力学模型验证。
  - **相关文献**：Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting" (ICLR 2018).
  - **参考实现**：[liyaguang/dcrnn](https://github.com/liyaguang/dcrnn)
  - **离线包下载**: [Google Drive - Common ST-Sequence](https://drive.google.com/file/d/1BKZ8Iqfo2x610sUxGEMCjpFYyn_UllTK/view?usp=sharing)
  - **处理流程与加载器**: [数据预处理与 Dataset 构建](./time_series/readme.md)

#### 6. PEMS-04 / PEMS-08
  - **描述**：加州高速公路交通流数据集。广泛应用于基于注意力机制的空间时空图卷积网络（ASTGCN）的基准测试。
  - **相关文献**：Guo et al., "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting" (AAAI 2019).
  - **参考实现**: [Davidham3/ASTGCN-2019-mxnet](https://github.com/Davidham3/ASTGCN-2019-mxnet)
  - **离线包下载**: [Google Drive - Common ST-Sequence](https://drive.google.com/file/d/1BKZ8Iqfo2x610sUxGEMCjpFYyn_UllTK/view?usp=sharing)
  - **处理流程与加载器**: [数据预处理与 Dataset 构建](./time_series/readme.md)

#### 7. TaxiBJ
  - **描述**：北京出租车轨迹数据集，包含网格化的流量数据及配套的气象、节假日信息，适合处理多源特征融合的时空预测任务。
  - **相关文献**：Zhang et al., "Deep Spatio-Temporal Residual Networks for Citywide Crowd Flow Prediction" (AAAI 2017).
  - **离线包下载**: [Google Drive - Common ST-Sequence](https://drive.google.com/file/d/1BKZ8Iqfo2x610sUxGEMCjpFYyn_UllTK/view?usp=sharing)
  - **处理流程与加载器**: [数据预处理与 Dataset 构建](./time_series/readme.md)

-----

### 📈 时间序列预测数据集 (Time-Series Forecasting)

#### 8. ETT (Electricity Transformer Temperature)
  - **描述**：包含变压器油温及 6 种电力负荷特征。分为 15 分钟（ETTm）和 1 小时（ETTh）两种粒度，是长时序列预测（LSTF）的基准数据。
  - **相关文献**：Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021 Best Paper).
  - **代码来源**: [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
  - **离线包下载**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **处理流程与加载器**: [数据预处理与 Dataset 构建](./time_series/readme.md)

#### 9. Electricity / Traffic
  - **描述**：涵盖电力消耗（kWh）及道路占用率（Occupancy）指标的通用时间序列数据集，常用于多变量长程预测任务。
  - **相关文献**：Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" (SIGIR 2018).
  - **代码来源**：[laiguokun/LSTNet](https://github.com/laiguokun/LSTNet)
  - **离线包下载**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **处理流程与加载器**: [数据预处理与 Dataset 构建](./time_series/readme.md)

#### 10. Weather
  - **描述**：包含气温、湿度、气压等 21 项气象指标的通用时间序列数据集，数据源自 MPI-BGC 气象站。
  - **官方数据源**: [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/)
  - **离线包下载**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **处理流程与加载器**: [数据预处理与 Dataset 构建](./time_series/readme.md)

#### 11. Exchange Rate
  - **描述**：包含 8 个国家（澳大利亚、英国、加拿大等）从 1990 年到 2016 年的每日汇率变动监测数据。
  - **相关文献**：Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" (SIGIR 2018).
  - **代码来源**：[laiguokun/LSTNet](https://github.com/laiguokun/LSTNet)
  - **离线包下载**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **处理流程与加载器**: [数据预处理与 Dataset 构建](./time_series/readme.md)

#### 12. Illness (ILI)
  - **描述**：来自美国疾控中心（CDC）的每周流感样病例（ILI）监测数据，包含门诊患者中 ILI 患者的占比，是衡量流行病预测模型长程建模能力的经典任务。
  - **官方数据源**：[CDC FluView Portal](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)
  - **离线包下载**: [Common Time-Series Package](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing)
  - **处理流程与加载器**: [数据预处理与 Dataset 构建](./time_series/readme.md)

-----

## <a id="download-summary"></a> ⬇️ 数据一键下载汇总

| 数据集分类 | 快速下载地址 (G-Drive) |
| :--- | :--- |
| **ERA5-Global 预处理二进制包** | [Download](https://drive.google.com/file/d/1_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view) |
| **通用时空序列数据集 (ST-Sequence)** | [Download](https://drive.google.com/file/d/1BKZ8Iqfo2x610sUxGEMCjpFYyn_UllTK/view?usp=sharing) |
| **通用时间序列基准 (Time-Series)** | [Download](https://drive.google.com/file/d/1qo-EWkPz-13IjYly9J_4B9In-aXuB--y/view?usp=sharing) |

-----

## 🤝 贡献与反馈

如果你在使用过程中发现代码 Bug，或者希望收录新的数据集，欢迎提交 **Issue** 或 **Pull Request**。