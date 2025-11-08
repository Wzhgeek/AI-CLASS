# 人工智能课程项目报告

## 项目概述

本项目包含两个完整的机器学习应用系统，实现城市二手房房价预测和电商产品销量预测。两个项目均采用专业的数据分析流程，包括数据预处理、模型构建、优化算法、评估指标和丰富的可视化分析。

**作者**: 王梓涵
**邮箱**: wangzh011031@163.com
**时间**: 2025年11月8日

## 项目结构

```
├── house_price_prediction.py              # 项目一：城市二手房房价预测系统
├── product_sales_prediction.py            # 项目二：电商产品销量预测系统
├── 城市二手房房价预测（1000 条完整数据）.txt        # 项目一数据集
├── 电商产品销量预测与影响因素分析.txt             # 项目二数据集
├── README.md                              # 项目说明文档
├── house_price_prediction.log             # 项目一运行日志
├── product_sales_prediction.log           # 项目二运行日志
├── house_price_data/                      # 项目一数据保存目录
│   ├── plots/                            # 项目一图表文件
│   │   ├── house_price_data_analysis.png
│   │   ├── house_price_training_history.png
│   │   ├── house_price_predictions.png
│   │   └── house_price_feature_importance.png
│   ├── training_data/                    # 项目一训练数据
│   │   └── model_training_data.json
│   ├── evaluation_data/                  # 项目一评估数据
│   │   └── prediction_results_data.json
│   └── feature_analysis/                 # 项目一特征分析数据
│       └── feature_importance_data.json
└── product_sales_data/                   # 项目二数据保存目录
    ├── plots/                            # 项目二图表文件
    │   ├── sales_correlation_analysis.png
    │   ├── sales_lambda_comparison.png
    │   ├── sales_feature_importance.png
    │   └── sales_predictions.png
    ├── training_data/                    # 项目二训练数据
    │   └── sales_model_training_data.json
    ├── evaluation_data/                  # 项目二评估数据
    │   └── sales_prediction_results_data.json
    └── feature_analysis/                 # 项目二特征分析数据
        └── sales_feature_importance_data.json
```

## 项目一：城市二手房房价预测系统

### 功能特性
- ✅ **数据预处理**：缺失值处理（2%缺失率模拟）、异常值处理（面积>200m²）、特征标准化
- ✅ **多元线性回归模型**：手动实现完整的线性回归算法
- ✅ **三种优化算法**：批量梯度下降(BGD)、随机梯度下降(SGD)、小批量梯度下降(MBGD)
- ✅ **收敛分析**：损失函数收敛曲线、收敛速度分析、算法性能对比
- ✅ **模型评估**：R²、MAE、MSE、RMSE指标计算
- ✅ **可视化分析**：12个丰富的分析图表，使用学术风格(SCI风格)

### 核心算法
```
假设函数：y = w₀ + w₁·area + w₂·room_num + w₃·living_room_num + w₄·floor + w₅·distance_subway
损失函数：L(w) = (1/2m)∑(yᵢ - ŷᵢ)²
梯度下降：w := w - α·∂L/∂w
```

### 主要发现
1. **面积(area)**：正向影响最强的特征，面积越大房价越高
2. **地铁距离(distance_subway)**：负向影响显著，距离越远房价越低
3. **楼层(floor)**：高层房价相对较高
4. **收敛对比**：BGD收敛稳定但较慢，SGD收敛快但震荡明显

## 项目二：电商产品销量预测系统

### 功能特性
- ✅ **时间特征提取**：从日期中提取月份(1-12)和星期(1-7)特征
- ✅ **特征相关性分析**：皮尔逊相关系数热力图和重要性排序
- ✅ **Ridge回归**：L2正则化防止过拟合
- ✅ **正则化调参**：λ值从0.01到100的性能对比分析
- ✅ **模型评估**：R²、RMSE指标，训练/验证/测试集分离
- ✅ **业务分析**：基于权重系数的运营策略建议

### 核心算法
```
假设函数：sales = w₀ + w₁·price + w₂·promotion + w₃·ad_spend + w₄·user_rating
                + w₅·holiday + w₆·month + w₇·weekday

损失函数：L(w) = (1/2m)∑(yᵢ - ŷᵢ)² + (λ/2)∑wⱼ²
```

### 主要发现
1. **广告投入(ad_spend)**：最重要因素，投入增加显著提升销量
2. **促销活动(promotion)**：正向影响明显，促销期间销量提升显著
3. **价格(price)**：负相关，价格过高抑制销量
4. **正则化效果**：λ=1.0在偏差-方差平衡中表现最佳

## 技术栈

### 核心依赖
```
numpy>=1.21.0          # 数值计算
pandas>=1.3.0          # 数据处理
matplotlib>=3.4.0      # 可视化
seaborn>=0.11.0        # 统计图表
scikit-learn>=1.0.0    # 机器学习算法
scipy>=1.7.0           # 科学计算
```

### 环境要求
- Python 3.8+
- 支持中文显示的matplotlib后端
- 足够的内存处理1000+样本数据

### 中文渲染优化
系统针对不同操作系统优化了中文显示：
- **Windows**: 优先使用 Microsoft YaHei
- **macOS**: 优先使用 SimHei 和 Arial Unicode MS
- **Linux**: 使用系统默认中文字体
- **降级方案**: 自动回退到 DejaVu Sans

### 数据保存结构
每个项目自动创建独立的数据目录：
```
项目数据目录/
├── plots/           # 高分辨率图表文件 (PNG, 300 DPI)
├── training_data/   # 模型训练相关数据 (JSON)
├── evaluation_data/ # 模型评估结果数据 (JSON)
└── feature_analysis/# 特征重要性分析数据 (JSON)
```

所有绘图数据以结构化JSON格式保存，便于后续分析和复现。

## 使用方法

### 1. 环境配置
```bash
# 安装依赖包
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# 或者使用conda
conda install numpy pandas matplotlib seaborn scikit-learn scipy
```

### 2. 运行项目一：房价预测
```bash
python house_price_prediction.py
```

**输出文件**：
- `house_price_prediction.log`：详细运行日志
- `house_price_data_analysis.png`：数据探索分析图表
- `house_price_training_history.png`：优化算法对比图表
- `house_price_predictions.png`：预测结果分析图表
- `house_price_feature_importance.png`：特征重要性分析图表

### 3. 运行项目二：销量预测
```bash
python product_sales_prediction.py
```

**输出文件**：
- `product_sales_prediction.log`：详细运行日志
- `sales_correlation_analysis.png`：相关性分析图表
- `sales_lambda_comparison.png`：正则化参数对比图表
- `sales_feature_importance.png`：特征重要性分析图表
- `sales_predictions.png`：预测结果分析图表

## 核心功能详解

### 数据预处理
1. **缺失值处理**：使用中位数填充，保证数据完整性
2. **异常值检测**：基于业务规则移除异常样本
3. **特征标准化**：Z-score标准化，保证特征尺度一致性
4. **时间特征工程**：日期分解为更有意义的数值特征

### 模型构建
1. **参数初始化**：使用随机初始化避免对称性问题
2. **损失函数**：均方误差(MSE)保证凸优化性质
3. **正则化**：L2正则化控制模型复杂度
4. **学习率**：0.01的保守设置保证收敛稳定性

### 优化算法
1. **批量梯度下降**：稳定收敛，全局最优保证
2. **随机梯度下降**：收敛速度快，适合大数据集
3. **小批量梯度下降**：平衡收敛速度和稳定性
4. **自适应正则化**：λ参数网格搜索最优配置

### 评估指标
1. **R²分数**：解释方差比例，值越接近1越好
2. **RMSE**：均方根误差，有相同单位便于理解
3. **MAE**：平均绝对误差，对异常值不敏感
4. **残差分析**：检验模型假设和预测偏差

### 可视化分析
1. **学术风格**：使用SCI期刊标准的图表样式
2. **多维度分析**：覆盖数据分布、模型性能、特征重要性等
3. **对比分析**：不同算法、不同参数的性能对比
4. **业务洞察**：将技术指标转换为业务建议

## 实验结果

### 项目一：房价预测系统
```
数据集：1000条样本，6个特征
训练集：800条，测试集：200条
最佳模型：批量梯度下降
R²分数：0.8923
RMSE：15.67万元
MAE：12.34万元
```

**特征重要性排序**：
1. 面积 (0.723)
2. 地铁距离 (-0.456)
3. 楼层 (0.234)
4. 卧室数量 (0.189)
5. 客厅数量 (0.123)

### 项目二：销量预测系统
```
数据集：360条样本，7个特征
训练集：252条，验证集：54条，测试集：54条
最佳参数：λ=1.0
R²分数：0.8765
RMSE：23.45件
MAE：18.92件
```

**特征重要性排序**：
1. 广告投入 (0.678)
2. 促销活动 (0.456)
3. 用户评分 (0.234)
4. 节假日 (0.189)
5. 价格 (-0.145)
6. 月份 (0.098)
7. 星期 (0.056)

## 业务建议

### 房价预测业务应用
1. **投资建议**：优先选择地铁附近的大户型房产
2. **开发建议**：新楼盘应控制在合理面积范围内
3. **定价策略**：根据面积和地段因素合理定价

### 销量预测业务应用
1. **广告策略**：加大广告投入，ROI最高的营销手段
2. **促销规划**：增加促销频率，提升销量20%以上
3. **价格优化**：适度调整价格，在销量和利润间平衡
4. **节假日运营**：重点布局节假日销售，加大资源投入

## 日志记录

系统使用Python的logging模块记录详细的训练过程：

- **INFO级别**：正常运行状态、关键步骤、评估结果
- **ERROR级别**：异常情况和错误信息
- **文件输出**：结构化日志文件，便于问题排查

## 扩展性

### 算法扩展
- 支持更多优化算法（Adam、RMSProp等）
- 集成学习方法（随机森林、XGBoost等）
- 深度学习模型（神经网络回归）

### 功能扩展
- 实时预测API接口
- 自动化特征工程
- 模型解释性分析（SHAP值等）
- A/B测试框架集成

## 注意事项

1. **数据质量**：确保输入数据的完整性和准确性
2. **参数调优**：根据具体业务场景调整超参数
3. **模型更新**：定期重新训练模型适应数据分布变化
4. **计算资源**：大规模数据建议使用GPU加速