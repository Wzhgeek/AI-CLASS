#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市二手房房价预测系统
项目一：基于多元线性回归的城市二手房房价预测

作者: 王梓涵
邮箱: wangzh011031@163.com
时间: 2025年11月8日

功能：
- 数据预处理（缺失值处理、异常值处理、特征标准化）
- 多元线性回归模型构建
- 批量梯度下降和随机梯度下降算法实现
- 模型评估和可视化
- 特征重要性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 设置Seaborn样式为学术风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('house_price_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HousePricePredictor:
    """
    城市二手房房价预测器

    实现多元线性回归模型，包含数据预处理、模型训练、优化算法和评估功能。
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=42):
        """
        初始化预测器

        参数:
        learning_rate (float): 梯度下降学习率
        max_iter (int): 最大迭代次数
        random_state (int): 随机种子
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names = ['area', 'room_num', 'living_room_num', 'floor', 'distance_subway']

        # 创建数据保存目录
        self.data_dir = "house_price_data"
        self.create_data_directories()

        logger.info("房价预测器初始化完成")

    def create_data_directories(self):
        """创建数据保存目录结构"""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, "plots"),
            os.path.join(self.data_dir, "training_data"),
            os.path.join(self.data_dir, "evaluation_data"),
            os.path.join(self.data_dir, "feature_analysis")
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")

    def save_plot_data(self, data_dict, filename):
        """保存绘图数据到JSON文件"""
        filepath = os.path.join(self.data_dir, filename)
        # 将numpy数组转换为列表以便JSON序列化
        serializable_data = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, (int, float, str, list)):
                serializable_data[key] = value
            else:
                serializable_data[key] = str(value)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        logger.info(f"绘图数据已保存至: {filepath}")

    def load_data(self, file_path):
        """
        加载房价数据集

        参数:
        file_path (str): 数据文件路径

        返回:
        pd.DataFrame: 加载的数据框
        """
        try:
            # 读取原始文件并处理注释
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 过滤掉空行和注释行，处理行内注释
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 处理行内注释
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    clean_lines.append(line)

            # 将清理后的数据写入临时字符串
            data_str = '\n'.join(clean_lines)

            # 使用StringIO读取数据
            from io import StringIO
            df = pd.read_csv(StringIO(data_str))

            # 确保所有数值列都是数值类型
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"成功加载数据，样本数量: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def preprocess_data(self, df, missing_rate=0.02, outlier_threshold=200):
        """
        数据预处理：缺失值处理、异常值处理、特征标准化

        参数:
        df (pd.DataFrame): 原始数据框
        missing_rate (float): 缺失率阈值
        outlier_threshold (float): 异常值面积阈值

        返回:
        tuple: (X_train, X_test, y_train, y_test, X_scaled, y_scaled)
        """
        logger.info("开始数据预处理...")

        # 1. 缺失值处理
        df_processed = self._handle_missing_values(df, missing_rate)
        logger.info(f"缺失值处理完成，当前样本数: {len(df_processed)}")

        # 2. 异常值处理
        df_processed = self._handle_outliers(df_processed, outlier_threshold)
        logger.info(f"异常值处理完成，当前样本数: {len(df_processed)}")

        # 3. 特征提取和标准化
        X = df_processed[self.feature_names].values
        y = df_processed['price'].values.reshape(-1, 1)

        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # 特征标准化
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # 目标变量标准化（用于训练）
        y_train_scaled = self.target_scaler.fit_transform(y_train)
        y_test_scaled = self.target_scaler.transform(y_test)

        logger.info("数据预处理完成")
        logger.info(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def _handle_missing_values(self, df, missing_rate):
        """处理缺失值"""
        # 随机选择2%的样本设置为缺失值（模拟）
        np.random.seed(self.random_state)
        n_missing = int(len(df) * missing_rate)
        missing_indices = np.random.choice(len(df), n_missing, replace=False)

        df_with_missing = df.copy()
        for idx in missing_indices:
            # 随机选择一个特征列设置为空值
            col = np.random.choice(self.feature_names)
            df_with_missing.loc[idx, col] = np.nan

        logger.info(f"模拟设置了 {n_missing} 个缺失值")

        # 用中位数填充缺失值
        for col in self.feature_names + ['price']:
            median_val = df_with_missing[col].median()
            df_with_missing[col] = df_with_missing[col].fillna(median_val)

        return df_with_missing

    def _handle_outliers(self, df, threshold):
        """处理异常值：移除面积大于阈值的样本"""
        original_count = len(df)
        df_cleaned = df[df['area'] <= threshold].copy()
        removed_count = original_count - len(df_cleaned)

        logger.info(f"移除了 {removed_count} 个异常值样本（面积 > {threshold}m²）")
        return df_cleaned

    def initialize_parameters(self, n_features):
        """初始化模型参数"""
        np.random.seed(self.random_state)
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0.0

        logger.info(f"参数初始化完成，权重维度: {self.weights.shape}")

    def hypothesis(self, X):
        """假设函数：线性回归预测"""
        return np.dot(X, self.weights) + self.bias

    def compute_cost(self, X, y):
        """计算均方误差损失函数"""
        m = len(y)
        predictions = self.hypothesis(X)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def gradient_descent(self, X, y, method='batch', batch_size=None):
        """
        梯度下降优化算法

        参数:
        X (np.array): 特征矩阵
        y (np.array): 目标变量
        method (str): 优化方法 ('batch', 'stochastic', 'mini_batch')
        batch_size (int): 小批量大小（仅用于mini_batch）

        返回:
        dict: 训练历史记录
        """
        logger.info(f"开始{method}梯度下降训练...")

        m, n = X.shape
        self.initialize_parameters(n)

        costs = []
        learning_rates = []

        for iteration in range(self.max_iter):
            if method == 'batch':
                # 批量梯度下降
                predictions = self.hypothesis(X)
                dw = (1 / m) * np.dot(X.T, (predictions - y))
                db = (1 / m) * np.sum(predictions - y)

            elif method == 'stochastic':
                # 随机梯度下降
                random_index = np.random.randint(m)
                x_i = X[random_index:random_index+1]
                y_i = y[random_index:random_index+1]

                prediction = self.hypothesis(x_i)
                dw = np.dot(x_i.T, (prediction - y_i))
                db = np.sum(prediction - y_i)

            elif method == 'mini_batch':
                # 小批量梯度下降
                if batch_size is None:
                    batch_size = min(32, m)

                indices = np.random.choice(m, batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]

                predictions = self.hypothesis(X_batch)
                dw = (1 / batch_size) * np.dot(X_batch.T, (predictions - y_batch))
                db = (1 / batch_size) * np.sum(predictions - y_batch)

            # 参数更新
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 记录损失
            if iteration % 100 == 0:
                cost = self.compute_cost(X, y)
                costs.append(cost)
                learning_rates.append(iteration)

                if iteration % 500 == 0:
                    logger.info(f"迭代 {iteration}, 损失: {cost:.6f}")

        logger.info(f"{method}梯度下降训练完成，最终损失: {costs[-1]:.6f}")
        return {'costs': costs, 'iterations': learning_rates}

    def predict(self, X):
        """预测房价"""
        X_scaled = self.feature_scaler.transform(X)
        return self.hypothesis(X_scaled)

    def evaluate_model(self, X_test, y_test, y_pred):
        """评估模型性能"""
        # 反标准化预测值以获得真实价格
        y_pred_original = self.target_scaler.inverse_transform(y_pred)

        mse = mean_squared_error(y_test, y_pred_original)
        mae = mean_absolute_error(y_test, y_pred_original)
        r2 = r2_score(y_test, y_pred_original)
        rmse = np.sqrt(mse)

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }

        logger.info("模型评估结果:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return metrics

    def plot_training_history(self, histories, save_path=None):
        """绘制训练历史曲线"""
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange']
        methods = list(histories.keys())

        # 准备保存的数据
        plot_data = {
            'methods': methods,
            'colors': colors,
            'training_histories': histories
        }

        for i, (method, history) in enumerate(histories.items()):
            plt.subplot(2, 2, 1)
            plt.plot(history['iterations'], history['costs'],
                    color=colors[i % len(colors)], label=method, linewidth=2)
            plt.xlabel('迭代次数')
            plt.ylabel('损失值')
            plt.title('不同优化算法的收敛曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 收敛速度分析
            plt.subplot(2, 2, 2)
            if len(history['costs']) > 1:
                convergence_rate = -np.diff(np.log(history['costs']))
                plt.plot(history['iterations'][1:], convergence_rate,
                        color=colors[i % len(colors)], label=method, linewidth=2)
                plot_data[f'{method}_convergence_rate'] = convergence_rate.tolist()
            plt.xlabel('迭代次数')
            plt.ylabel('收敛速度')
            plt.title('收敛速度分析')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片和数据
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图表已保存至: {save_path}")

        # 保存数据到plots目录
        data_filename = "training_history_data.json"
        self.save_plot_data(plot_data, os.path.join("plots", data_filename))

        plt.close()  # 关闭图形

    def plot_predictions(self, y_test, y_pred, save_path=None):
        """绘制预测结果对比图"""
        # 反标准化预测值
        y_pred_original = self.target_scaler.inverse_transform(y_pred)

        plt.figure(figsize=(15, 10))

        # 计算各种数据用于保存
        residuals = y_test.flatten() - y_pred_original.flatten()
        relative_errors = np.abs(residuals / y_test.flatten()) * 100

        # 准备保存的数据
        plot_data = {
            'y_test': y_test.flatten(),
            'y_pred_original': y_pred_original.flatten(),
            'residuals': residuals,
            'relative_errors': relative_errors,
            'test_min': float(y_test.min()),
            'test_max': float(y_test.max())
        }

        # 散点图：预测值vs真实值
        plt.subplot(2, 3, 1)
        plt.scatter(y_test, y_pred_original, alpha=0.6, color='blue', edgecolors='black')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='完美预测线')
        plt.xlabel('真实房价 (万元)')
        plt.ylabel('预测房价 (万元)')
        plt.title('预测值 vs 真实值散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 残差图
        plt.subplot(2, 3, 2)
        plt.scatter(y_pred_original, residuals, alpha=0.6, color='green', edgecolors='black')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('预测房价 (万元)')
        plt.ylabel('残差 (万元)')
        plt.title('残差分布图')
        plt.grid(True, alpha=0.3)

        # 残差直方图
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('残差 (万元)')
        plt.ylabel('频次')
        plt.title('残差直方图')
        plt.grid(True, alpha=0.3)

        # Q-Q图
        plt.subplot(2, 3, 4)
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        plt.plot(osm, osr, 'o', alpha=0.6, color='blue')
        plt.plot(osm, slope*osm + intercept, 'r--', linewidth=2)
        plt.xlabel('理论分位数')
        plt.ylabel('样本分位数')
        plt.title('残差 Q-Q 图')
        plt.grid(True, alpha=0.3)

        plot_data['qq_plot'] = {
            'osm': osm.tolist(),
            'osr': osr.tolist(),
            'slope': float(slope),
            'intercept': float(intercept),
            'r': float(r)
        }

        # 预测误差分布
        plt.subplot(2, 3, 5)
        plt.hist(y_pred_original.flatten(), bins=30, alpha=0.7, color='purple', edgecolor='black', label='预测值')
        plt.hist(y_test.flatten(), bins=30, alpha=0.5, color='cyan', edgecolor='black', label='真实值')
        plt.xlabel('房价 (万元)')
        plt.ylabel('频次')
        plt.title('预测值与真实值分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 相对误差分布
        plt.subplot(2, 3, 6)
        plt.hist(relative_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('相对误差 (%)')
        plt.ylabel('频次')
        plt.title('相对误差分布')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片和数据
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图表已保存至: {save_path}")

        # 保存数据到evaluation_data目录
        data_filename = "prediction_results_data.json"
        self.save_plot_data(plot_data, os.path.join("evaluation_data", data_filename))

        plt.close()  # 关闭图形

    def analyze_feature_importance(self, feature_names, save_path=None):
        """分析特征重要性"""
        plt.figure(figsize=(12, 8))

        weights = self.weights.flatten()
        abs_weights = np.abs(weights)
        colors = ['red' if w > 0 else 'blue' for w in weights]
        importance_percent = (abs_weights / abs_weights.sum()) * 100
        sorted_idx = np.argsort(abs_weights)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_weights = abs_weights[sorted_idx]

        # 准备保存的数据
        plot_data = {
            'feature_names': feature_names,
            'weights': weights.tolist(),
            'abs_weights': abs_weights.tolist(),
            'colors': colors,
            'importance_percent': importance_percent.tolist(),
            'sorted_features': sorted_features,
            'sorted_weights': sorted_weights.tolist()
        }

        # 特征权重分析
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(weights)), weights, color=colors, alpha=0.7, edgecolor='black')
        plt.xticks(range(len(weights)), feature_names, rotation=45, ha='right')
        plt.xlabel('特征')
        plt.ylabel('权重系数')
        plt.title('特征权重系数分析')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, weight in zip(bars, weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    '.3f', ha='center', va='bottom' if weight > 0 else 'top')

        # 特征重要性百分比
        plt.subplot(2, 2, 2)
        plt.pie(importance_percent, labels=feature_names, autopct='%1.1f%%',
               startangle=90, colors=plt.cm.Set3.colors)
        plt.title('特征重要性百分比')
        plt.axis('equal')

        # 权重系数的绝对值排序
        plt.subplot(2, 2, 3)
        bars = plt.barh(range(len(sorted_features)), sorted_weights,
                       color='skyblue', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('权重绝对值')
        plt.ylabel('特征')
        plt.title('特征重要性排序（绝对值）')
        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # 保存图片和数据
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性分析图表已保存至: {save_path}")

        # 保存数据到feature_analysis目录
        data_filename = "feature_importance_data.json"
        self.save_plot_data(plot_data, os.path.join("feature_analysis", data_filename))

        plt.close()  # 关闭图形

        # 输出特征重要性分析结果
        logger.info("特征重要性分析:")
        for name, weight in zip(feature_names, weights):
            effect = "正向影响" if weight > 0 else "负向影响"
            logger.info(f"{name}: 权重={weight:.4f} ({effect})")

    def plot_data_analysis(self, df, save_path=None):
        """数据探索性分析可视化"""
        plt.figure(figsize=(16, 12))

        # 房价分布
        plt.subplot(3, 4, 1)
        plt.hist(df['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('房价 (万元)')
        plt.ylabel('频次')
        plt.title('房价分布直方图')
        plt.grid(True, alpha=0.3)

        # 面积分布
        plt.subplot(3, 4, 2)
        plt.hist(df['area'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('面积 (m²)')
        plt.ylabel('频次')
        plt.title('房屋面积分布')
        plt.grid(True, alpha=0.3)

        # 房价vs面积散点图
        plt.subplot(3, 4, 3)
        plt.scatter(df['area'], df['price'], alpha=0.6, color='coral', edgecolors='black')
        plt.xlabel('面积 (m²)')
        plt.ylabel('房价 (万元)')
        plt.title('房价 vs 面积')
        plt.grid(True, alpha=0.3)

        # 房价vs地铁距离散点图
        plt.subplot(3, 4, 4)
        plt.scatter(df['distance_subway'], df['price'], alpha=0.6, color='purple', edgecolors='black')
        plt.xlabel('地铁距离 (km)')
        plt.ylabel('房价 (万元)')
        plt.title('房价 vs 地铁距离')
        plt.grid(True, alpha=0.3)

        # 箱线图：不同卧室数量的房价分布
        plt.subplot(3, 4, 5)
        room_prices = [df[df['room_num'] == i]['price'].values for i in sorted(df['room_num'].unique())]
        plt.boxplot(room_prices, labels=sorted(df['room_num'].unique()))
        plt.xlabel('卧室数量')
        plt.ylabel('房价 (万元)')
        plt.title('不同卧室数量的房价分布')
        plt.grid(True, alpha=0.3)

        # 箱线图：不同楼层的房价分布
        plt.subplot(3, 4, 6)
        floor_prices = []
        floor_labels = []
        for floor in sorted(df['floor'].unique())[:10]:  # 只显示前10个楼层
            floor_prices.append(df[df['floor'] == floor]['price'].values)
            floor_labels.append(str(floor))

        plt.boxplot(floor_prices, labels=floor_labels)
        plt.xlabel('楼层')
        plt.ylabel('房价 (万元)')
        plt.title('不同楼层的房价分布')
        plt.grid(True, alpha=0.3)

        # 相关性热力图
        plt.subplot(3, 4, 7)
        corr_matrix = df[self.feature_names + ['price']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('特征相关性热力图')

        # 房价vs楼层散点图
        plt.subplot(3, 4, 8)
        plt.scatter(df['floor'], df['price'], alpha=0.6, color='orange', edgecolors='black')
        plt.xlabel('楼层')
        plt.ylabel('房价 (万元)')
        plt.title('房价 vs 楼层')
        plt.grid(True, alpha=0.3)

        # 特征间的散点图矩阵（简化版）
        plt.subplot(3, 4, 9)
        selected_features = ['area', 'distance_subway', 'price']
        sns.pairplot(df[selected_features], kind='scatter', diag_kind='hist')
        plt.suptitle('主要特征散点图矩阵', y=0.95)

        # 房价统计信息
        plt.subplot(3, 4, 10)
        plt.text(0.1, 0.8, f'样本总数: {len(df)}', fontsize=12)
        plt.text(0.1, 0.7, f'平均房价: {df["price"].mean():.1f}万元', fontsize=12)
        plt.text(0.1, 0.6, f'房价中位数: {df["price"].median():.1f}万元', fontsize=12)
        plt.text(0.1, 0.5, f'房价标准差: {df["price"].std():.1f}万元', fontsize=12)
        plt.text(0.1, 0.4, f'房价最小值: {df["price"].min():.1f}万元', fontsize=12)
        plt.text(0.1, 0.3, f'房价最大值: {df["price"].max():.1f}万元', fontsize=12)
        plt.title('房价统计信息')
        plt.axis('off')

        # 房价分位数分布
        plt.subplot(3, 4, 11)
        quantiles = [0.25, 0.5, 0.75, 0.9, 0.95]
        quantile_values = np.quantile(df['price'], quantiles)
        plt.bar(range(len(quantiles)), quantile_values,
               color='lightblue', alpha=0.7, edgecolor='black')
        plt.xticks(range(len(quantiles)), [f'{q*100}%' for q in quantiles])
        plt.ylabel('房价 (万元)')
        plt.title('房价分位数分布')
        plt.grid(True, alpha=0.3, axis='y')

        # 面积分位数分布
        plt.subplot(3, 4, 12)
        area_quantiles = np.quantile(df['area'], quantiles)
        plt.bar(range(len(quantiles)), area_quantiles,
               color='lightcoral', alpha=0.7, edgecolor='black')
        plt.xticks(range(len(quantiles)), [f'{q*100}%' for q in quantiles])
        plt.ylabel('面积 (m²)')
        plt.title('面积分位数分布')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"数据分析图表已保存至: {save_path}")
        plt.show()


def main():
    """主函数：运行完整的房价预测系统"""
    logger.info("开始运行城市二手房房价预测系统")

    # 初始化预测器
    predictor = HousePricePredictor(learning_rate=0.01, max_iter=1000)

    try:
        # 1. 加载数据
        df = predictor.load_data('城市二手房房价预测（1000 条完整数据）.txt')

        # 2. 数据预处理
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = \
            predictor.preprocess_data(df)

        # 3. 数据探索性分析
        predictor.plot_data_analysis(df, save_path=os.path.join(predictor.data_dir, 'plots', 'house_price_data_analysis.png'))

        # 4. 训练不同优化算法的模型
        methods = ['batch', 'stochastic', 'mini_batch']
        histories = {}

        for method in methods:
            logger.info(f"训练 {method} 梯度下降模型...")
            if method == 'mini_batch':
                history = predictor.gradient_descent(X_train_scaled, y_train_scaled,
                                                   method=method, batch_size=32)
            else:
                history = predictor.gradient_descent(X_train_scaled, y_train_scaled, method=method)

            histories[method] = history

        # 5. 绘制训练历史
        predictor.plot_training_history(histories, save_path=os.path.join(predictor.data_dir, 'plots', 'house_price_training_history.png'))

        # 6. 使用批量梯度下降模型进行预测和评估
        predictor.initialize_parameters(X_train_scaled.shape[1])
        predictor.gradient_descent(X_train_scaled, y_train_scaled, method='batch')

        y_pred = predictor.predict(X_test)
        metrics = predictor.evaluate_model(X_test, y_test, y_pred)

        # 7. 预测结果可视化
        predictor.plot_predictions(y_test, y_pred, save_path=os.path.join(predictor.data_dir, 'plots', 'house_price_predictions.png'))

        # 8. 特征重要性分析
        predictor.analyze_feature_importance(predictor.feature_names,
                                           save_path=os.path.join(predictor.data_dir, 'plots', 'house_price_feature_importance.png'))

        # 保存训练数据
        training_data = {
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
            'y_train_shape': y_train.shape,
            'y_test_shape': y_test.shape,
            'feature_names': predictor.feature_names,
            'final_weights': predictor.weights.tolist(),
            'final_bias': float(predictor.bias),
            'metrics': metrics
        }
        predictor.save_plot_data(training_data, os.path.join("training_data", "model_training_data.json"))

        # 9. 输出最终结果
        logger.info("=" * 50)
        logger.info("城市二手房房价预测系统运行完成")
        logger.info("=" * 50)
        logger.info("模型评估指标:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info("特征重要性分析:")
        weights = predictor.weights.flatten()
        for name, weight in zip(predictor.feature_names, weights):
            effect = "正向影响房价" if weight > 0 else "负向影响房价"
            impact = "增加" if weight > 0 else "减少"
            logger.info(f"{name}: {effect}，权重系数为{weight:.4f}")

    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        raise


if __name__ == "__main__":
    main()
