# Scientific-Data-Pipeline

一个专为**科学计算**、**流体动力学**与**时空序列预测**设计的全流程数据集仓库。

<p align="center">
  <b>中文说明</b> | <a href="./readme.md">English</a>
</p>

> [!WARNING]
> **开发中**：仓库内容正在频繁更新，部分预处理脚本可能存在不稳定因素。如遇运行异常，欢迎提交 Issue。

-----

## 📖 快速导航

  - **[已收录数据集简介](https://www.google.com/search?q=%23%E5%B7%B2%E6%94%B6%E5%BD%95%E6%95%B0%E6%8D%AE%E9%9B%86)**：快速了解各数据集背景及适用场景。
  - **[数据集统计详情](https://www.google.com/search?q=./data_stats.md)**：查看样本量、特征维数等详细统计信息。
  - **[打包数据下载](https://www.google.com/search?q=%23%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD)**：直接获取处理好的二进制数据包。

-----

## 🌟 项目初衷 (Welcome)

在开展科学计算（如流体模拟、气象预测）相关的深度学习研究时，获取高质量的数据集往往是第一道门槛。

虽然公开平台很多，但个人研究者常面临以下痛点：

1.  **获取成本高**：大规模原始数据的下载、清洗与对齐极其耗时。
2.  **复现困难**：论文中私有的预处理逻辑导致实验结果难以对标。
3.  **工具链断裂**：从原始表格到 PyTorch `Dataset` 的转换过程枯燥且重复。

**Scientific-Data-Pipeline** 的存在不仅是为了提供数据，更是为了公开**从数据下载、预处理到构建 Torch Dataset 的完整代码流**。我们追求的是研究的**透明度**与**可用性**，旨在为计算数学与深度学习的交叉领域提供可靠的基础支撑。

> [!WARNING]
> **注**：本项目并非为了“刷榜”而建。如果你需要标准的时间序列 Benchmark 评测，推荐前往 [Time-Series-Library](https://github.com/thuml/Time-Series-Library)。

-----

## 📂 已收录数据集

### 🌍 气象重构与多尺度数据集 (Meteorological Datasets)

#### 1\. Shenzhen Automatic Weather Station Data (城市尺度)

  - **描述**：来自深圳市自动气象站的高密度区域监测数据。其节点分布极密，非常适合高保真空间插值任务以及构建复杂的图神经网络（如 GAT, GCN）邻接矩阵。
  - **来源**：深圳政府数据开放平台
  - **处理建议**：原始数据为表格形式，建议转换为 `.npy` 或 `.bin` 格式以提升加载效率。
  - **download**: [暂留，指向下载连接]
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

#### 2\. ERA5-CN (国家尺度)

  - **描述**：从全球 ERA5 再分析资料中提取的中国区域数据集。分辨率适中，是开展区域性时空预测、数据同化研究以及大规模模型预训练的理想选择。
  - **来源**：Copernicus Climate Data Store (CDS)
  - **download**: [暂留，指向下载连接]
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

#### 3\. ERA5-Global (全球尺度)

  - **描述**：覆盖全球的行星级数据集。对于训练全球天气预测大模型或为数值偏微分方程（PDE）求解器生成初始/边界条件至关重要。
  - **来源**：Copernicus Climate Data Store (CDS)
  - **download**: [https://drive.google.com/file/d/1\_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view](https://drive.google.com/file/d/1_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view)
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

-----

### 🚦 时空序列预测数据集 (Spatio-Temporal Forecasting)

#### 4\. METR-LA / PEMS-BAY

  - **描述**：经典的交通速度数据集，分别包含洛杉矶和湾区的传感器监测数据。常用于扩散卷积循环神经网络（DCRNN）等图动力学模型验证。
  - **download**: [暂留，指向下载连接]
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

#### 5\. PEMS-04 / PEMS-08

  - **描述**：加州高速公路交通流数据集。广泛应用于基于注意力机制的空间时空图卷积网络（ASTGCN）的 Benchmark 测试。
  - **download**: [暂留，指向下载连接]
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

#### 6\. TaxiBJ

  - **描述**：北京出租车轨迹数据集，包含网格化的流量数据及配套的气象、节假日信息，适合处理多源特征融合的时空预测任务。
  - **download**: [暂留，指向下载连接]
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

-----

### 📈 时间序列预测数据集 (Time-Series Forecasting)

#### 7\. ETT (Electricity Transformer Temperature)

  - **描述**：包含变压器油温及 6 种电力负荷特征。分为 15 分钟（ETTm）和 1 小时（ETTh）两种粒度，是长时序列预测的基准数据。
  - **download**: [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

#### 8\. Electricity / Traffic / Weather

  - **描述**：覆盖电力消耗、道路占用率及气象指标（气温、湿度等）的通用时间序列数据集。
  - **download**: [https://drive.google.com/file/d/1rHJYc8cgNFPPvWLRpwynGj2xohqcc2R7/view?usp=sharing](https://drive.google.com/file/d/1rHJYc8cgNFPPvWLRpwynGj2xohqcc2R7/view?usp=sharing)
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

#### 9\. Exchange Rate / Illness (ILI)

  - **描述**：包含多国汇率变动及 CDC 流感样病例监测数据。
  - **download**: [暂留，指向下载连接]
  - **Process Code**: [暂留，指向相应数据集下载、预处理的代码目录中的 readme 文件]
  - **供参考的 Dataset Class**: [暂留，指向相应目录]

-----

## ⬇️ 数据下载汇总

如果你希望快速开始实验，可以直接下载以下打包好的文件：

| 数据集分类 | 下载地址 |
| :--- | :--- |
| **ERA5-Global 二进制包** | [Google Drive](https://drive.google.com/file/d/1_USk4qPbhMNM3sB9mzfKAL3H-Z568V8Q/view) |
| **常用时间序列数据集** | [Google Drive](https://drive.google.com/file/d/1rHJYc8cgNFPPvWLRpwynGj2xohqcc2R7/view?usp=sharing) |

-----

## 🤝 贡献与反馈

如果你在使用过程中发现代码 Bug，或者希望收录新的数据集，欢迎提交 **Issue** 或 **Pull Request**。