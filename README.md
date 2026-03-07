# DL-test
# 面向树种分类的轻量化多模态全局-局部特征协同网络

## 项目介绍

本项目结合多源遥感影像（Sentinel-2 光学影像、Sentinel-1 雷达影像、高分二号高分辨率影像）与全球城市树种数据集(GUTS)，采用Swin Transformer+MLP多模态融合深度学习模型，实现郑州地区的树种精准分类，解决单一属性数据分类的局限性，提升分类的客观性和泛化能力。

## 依赖安装

执行以下命令安装所需依赖：

```bash

pip install pandas numpy tensorflow scikit-learn joblib rasterio matplotlib
```

## 使用方法

1. **数据准备**

    - 将遥感影像文件（Sentinel-2、Sentinel-1、GF-2）放入项目的`./data`目录

    - 将全球城市树种数据集(GUTS)（GUTS_dataset.xlsx）放入`./data`目录


2. **运行项目**
打开 Jupyter Notebook 并执行完整脚本：

    ```bash
    
    jupyter notebook tree_species_remote.ipynb
    ```

    按照 Notebook 中的步骤依次执行，完成数据预处理、模型训练、模型评估和样本预测。

## 文件结构

```Plain Text

├── data/                 # 数据目录
│   ├── sentinel2.tif     # Sentinel-2多光谱遥感影像
│   ├── sentinel1.tif     # Sentinel-1 SAR雷达影像
│   ├── gf2.tif           # 高分二号高分辨率影像
│   └── GUTS_dataset.xlsx # 全球城市树种数据集(GUTS)
├── tree_species_remote.ipynb # 项目主运行脚本
└── README.md             # 项目说明文档
```

## 模型说明

1. **双分支特征提取**

    - 遥感影像分支：使用 Swin Transformer 作为骨干网络，提取影像的空间长程依赖特征和光谱特征，加入 ASPP 多尺度特征增强模块

    - 属性数据分支：使用多层感知机（MLP）提取属性数据的特征，包含文献验证、物种属性等信息

2. **交叉注意力融合**

    - 采用交叉注意力模块融合两类模态的特征，动态调整遥感特征和属性特征的权重，提升融合效果

3. **损失函数优化**

    - 使用 SparseCategoricalFocalCrossentropy 损失函数，解决树种类别不平衡问题，提升小样本树种的识别精度

## 结果说明

- 模型在测试集上的准确率较高（最终精度取决于数据质量和样本量）

- 可输出郑州树种分布专题图、分类报告、混淆矩阵、Kappa 系数等评估结果

- 支持单个样本的树种预测，输出预测树种名称和预测概率

## 数据说明

- 遥感影像可从[哥白尼数据开放中心](https://scihub.copernicus.eu/)、[国家地理信息公共服务平台](https://www.tianditu.gov.cn/)获取

- 全球城市树种数据集(GUTS)来源于Yang, X., Yan, P., Jin, J.等.全球城市树种（GUTS）：揭示世界城市地区的树种多样性. 科学数据 (2026). https://doi.org/10.1038/s41597-026-06868-2
  
## 贡献

欢迎提交 PR 或 Issue，共同优化模型结构、提升分类精度，适配更多城市的树种分类场景。
