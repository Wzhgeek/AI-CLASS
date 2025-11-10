#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电商产品销量预测与影响因素分析
项目二：基于线性回归+L2正则化的电商产品销量预测系统

作者: 王梓涵
邮箱: wangzh011031@163.com
时间: 2025年11月8日

功能：
- 时间特征提取（月份、星期）
- 特征相关性分析
- 线性回归+L2正则化（Ridge回归）
- 不同正则化参数对比分析
- 模型评估和可视化
- 业务分析和运营建议
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置科研绘图风格
sns.set_style("whitegrid")  # 使用白色网格背景
sns.set_context("paper", font_scale=1.2)  # 使用论文风格，字体放大1.2倍
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 高分辨率
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('product_sales_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductSalesPredictor:
    """
    电商产品销量预测器

    实现基于Ridge回归的销量预测模型，包含时间特征提取、正则化调参和业务分析。
    """

    def __init__(self, random_state=42):
        """
        初始化预测器

        参数:
        random_state (int): 随机种子
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.best_model = None
        self.best_lambda = None
        self.feature_names = ['price', 'promotion', 'ad_spend', 'user_rating', 'holiday', 'month', 'weekday']

        # 创建数据保存目录
        self.data_dir = "product_sales_data"
        self.create_data_directories()

        logger.info("销量预测器初始化完成")

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
        加载销量数据集

        参数:
        file_path (str): 数据文件路径

        返回:
        pd.DataFrame: 加载的数据框
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"成功加载数据，样本数量: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def preprocess_data(self, df):
        """
        数据预处理：时间特征提取、特征编码

        参数:
        df (pd.DataFrame): 原始数据框

        返回:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        logger.info("开始数据预处理...")

        # 1. 时间特征提取
        df_processed = self._extract_time_features(df)

        # 2. 特征编码（离散特征已经是数值型，无需额外编码）
        # promotion和holiday已经是0/1编码

        # 3. 特征和目标变量分离
        feature_cols = ['price', 'promotion', 'ad_spend', 'user_rating', 'holiday', 'month', 'weekday']
        X = df_processed[feature_cols].values
        y = df_processed['sales'].values.reshape(-1, 1)

        # 4. 数据集划分：训练集70%，验证集15%，测试集15%
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=self.random_state  # 15% of total
        )

        logger.info("数据预处理完成")
        logger.info(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}, 测试集大小: {X_test.shape[0]}")

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def _extract_time_features(self, df):
        """提取时间特征：月份和星期"""
        df_processed = df.copy()

        # 将日期字符串转换为datetime对象
        df_processed['date'] = pd.to_datetime(df_processed['date'])

        # 提取月份特征 (1-12)
        df_processed['month'] = df_processed['date'].dt.month

        # 提取星期特征 (1-7, 1=星期一)
        df_processed['weekday'] = df_processed['date'].dt.dayofweek + 1

        logger.info("时间特征提取完成：月份和星期特征")
        return df_processed

    def analyze_correlation(self, df, save_path=None):
        """特征相关性分析"""
        plt.figure(figsize=(12, 10))

        # 计算相关性矩阵
        feature_cols = ['price', 'promotion', 'ad_spend', 'user_rating', 'holiday', 'month', 'weekday', 'sales']
        corr_matrix = df[feature_cols].corr()

        # 相关性热力图
        plt.subplot(2, 2, 1)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'shrink': 0.8}, fmt='.3f')
        plt.title('特征相关性热力图')

        # 与销量相关的特征重要性
        plt.subplot(2, 2, 2)
        sales_corr = corr_matrix['sales'].drop('sales').abs().sort_values(ascending=True)
        colors = ['lightcoral' if corr_matrix.loc[name, 'sales'] > 0 else 'lightblue'
                 for name in sales_corr.index]

        bars = plt.barh(range(len(sales_corr)), sales_corr.values, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(sales_corr)), sales_corr.index)
        plt.xlabel('相关系数绝对值')
        plt.ylabel('特征')
        plt.title('各特征与销量相关性')
        plt.grid(True, alpha=0.3, axis='x')

        # 添加相关系数数值
        for i, (name, corr) in enumerate(zip(sales_corr.index, sales_corr.values)):
            actual_corr = corr_matrix.loc[name, 'sales']
            plt.text(corr - 0.02, i, f'{actual_corr:.3f}', ha='right', va='center', fontsize=8)

        # 散点图：销量vs广告投入
        plt.subplot(2, 2, 3)
        plt.scatter(df['ad_spend'], df['sales'], alpha=0.6, color='green', edgecolors='black')
        plt.xlabel('广告投入 (元)')
        plt.ylabel('销量 (件)')
        plt.title('销量 vs 广告投入')
        plt.grid(True, alpha=0.3)

        # 箱线图：促销 vs 销量
        plt.subplot(2, 2, 4)
        promotion_sales = [df[df['promotion'] == 0]['sales'].values,
                          df[df['promotion'] == 1]['sales'].values]
        plt.boxplot(promotion_sales, labels=['无促销', '促销中'])
        plt.ylabel('销量 (件)')
        plt.title('促销对销量的影响')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"相关性分析图表已保存至: {save_path}")
        plt.close()

        # 输出相关性分析结果
        logger.info("特征相关性分析:")
        for feature in sales_corr.index[::-1]:  # 从高到低
            corr_value = corr_matrix.loc[feature, 'sales']
            direction = "正相关" if corr_value > 0 else "负相关"
            strength = "强" if abs(corr_value) > 0.5 else "中等" if abs(corr_value) > 0.3 else "弱"
            logger.info(f"{feature}: {direction} ({strength}相关), 相关系数={corr_value:.3f}")

    def train_ridge_models(self, X_train, y_train, X_val, y_val, lambda_values=None):
        """
        训练多个Ridge回归模型，使用不同正则化参数

        参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        lambda_values: 正则化参数列表

        返回:
        dict: 各模型的性能指标
        """
        if lambda_values is None:
            lambda_values = [0.01, 0.1, 1, 10, 100]

        logger.info("开始训练Ridge回归模型，不同λ值对比...")

        results = {}
        best_r2 = -np.inf

        for lambda_val in lambda_values:
            # 训练Ridge模型
            model = Ridge(alpha=lambda_val, random_state=self.random_state)
            model.fit(X_train, y_train.ravel())

            # 在验证集上评估
            y_val_pred = model.predict(X_val)
            r2_val = r2_score(y_val, y_val_pred)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

            results[lambda_val] = {
                'model': model,
                'r2_val': r2_val,
                'rmse_val': rmse_val,
                'weights': model.coef_,
                'intercept': model.intercept_
            }

            logger.info(f"λ={lambda_val}: 验证集R²={r2_val:.4f}, RMSE={rmse_val:.4f}")

            # 更新最佳模型
            if r2_val > best_r2:
                best_r2 = r2_val
                self.best_model = model
                self.best_lambda = lambda_val

        logger.info(f"最佳模型: λ={self.best_lambda}, 验证集R²={best_r2:.4f}")
        return results

    def evaluate_model(self, X_test, y_test):
        """在测试集上评估最佳模型"""
        y_pred = self.best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }

        logger.info("测试集评估结果:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return metrics, y_pred

    def plot_lambda_comparison(self, results, X_train, X_val, y_train, y_val, save_path=None):
        """绘制不同正则化参数的性能对比"""
        plt.figure(figsize=(15, 10))

        lambda_values = list(results.keys())
        r2_scores = [results[lam]['r2_val'] for lam in lambda_values]
        rmse_scores = [results[lam]['rmse_val'] for lam in lambda_values]

        # R²分数对比
        plt.subplot(2, 3, 1)
        bars = plt.bar(range(len(lambda_values)), r2_scores,
                      color='skyblue', alpha=0.7, edgecolor='black')
        plt.xticks(range(len(lambda_values)), [f'{lam}' for lam in lambda_values])
        plt.xlabel('正则化参数 λ')
        plt.ylabel('验证集R²分数')
        plt.title('不同λ值的R²分数对比')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=8)

        # RMSE对比
        plt.subplot(2, 3, 2)
        bars = plt.bar(range(len(lambda_values)), rmse_scores,
                      color='lightcoral', alpha=0.7, edgecolor='black')
        plt.xticks(range(len(lambda_values)), [f'{lam}' for lam in lambda_values])
        plt.xlabel('正则化参数 λ')
        plt.ylabel('验证集RMSE')
        plt.title('不同λ值的RMSE对比')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, score in zip(bars, rmse_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=8)

        # 计算权重矩阵
        weights_matrix = np.array([results[lam]['weights'] for lam in lambda_values])

        # 准备保存的数据
        plot_data = {
            'lambda_values': lambda_values,
            'r2_scores': r2_scores,
            'rmse_scores': rmse_scores,
            'weights_matrix': weights_matrix.tolist(),
            'best_lambda': self.best_lambda
        }

        # 权重系数随λ变化
        plt.subplot(2, 3, 3)

        for i, feature in enumerate(self.feature_names):
            plt.plot(lambda_values, weights_matrix[:, i], 'o-', label=feature, linewidth=2, markersize=6)

        plt.xscale('log')
        plt.xlabel('正则化参数 λ')
        plt.ylabel('权重系数')
        plt.title('权重系数随λ的变化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # 模型复杂度 vs 性能
        plt.subplot(2, 3, 4)
        # 计算模型复杂度（权重绝对值之和）
        complexities = [np.sum(np.abs(results[lam]['weights'])) for lam in lambda_values]

        plt.scatter(complexities, r2_scores, s=100, c=lambda_values,
                   cmap='viridis', alpha=0.7, edgecolors='black')
        plt.colorbar(label='λ值')
        plt.xlabel('模型复杂度 (权重绝对值之和)')
        plt.ylabel('验证集R²分数')
        plt.title('模型复杂度 vs 性能')
        plt.grid(True, alpha=0.3)

        # 残差分析
        plt.subplot(2, 3, 5)
        best_lambda = self.best_lambda
        best_model = results[best_lambda]['model']

        # 使用训练数据进行残差分析
        X_all = np.vstack([X_train, X_val])
        y_all = np.vstack([y_train, y_val])
        y_all_pred = best_model.predict(X_all)
        residuals = y_all.flatten() - y_all_pred

        plt.scatter(y_all_pred, residuals, alpha=0.6, color='green', edgecolors='black')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('预测销量 (件)')
        plt.ylabel('残差 (件)')
        plt.title(f'最佳模型残差分析 (λ={best_lambda})')
        plt.grid(True, alpha=0.3)

        # 预测误差分布
        plt.subplot(2, 3, 6)
        plt.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('残差 (件)')
        plt.ylabel('频次')
        plt.title('预测误差分布')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片和数据
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"λ值对比分析图表已保存至: {save_path}")

        # 保存数据到plots目录
        data_filename = "lambda_comparison_data.json"
        self.save_plot_data(plot_data, os.path.join("plots", data_filename))

        plt.close()  # 关闭图形

    def plot_predictions(self, y_test, y_pred, save_path=None):
        """绘制预测结果对比图"""
        plt.figure(figsize=(15, 10))

        # 散点图：预测值vs真实值
        plt.subplot(2, 3, 1)
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='完美预测线')
        plt.xlabel('真实销量 (件)')
        plt.ylabel('预测销量 (件)')
        plt.title('预测值 vs 真实值散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 残差图
        plt.subplot(2, 3, 2)
        residuals = y_test.flatten() - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('预测销量 (件)')
        plt.ylabel('残差 (件)')
        plt.title('残差分布图')
        plt.grid(True, alpha=0.3)

        # 残差直方图
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('残差 (件)')
        plt.ylabel('频次')
        plt.title('残差直方图')
        plt.grid(True, alpha=0.3)

        # 预测值vs真实值分布对比
        plt.subplot(2, 3, 4)
        plt.hist(y_pred, bins=20, alpha=0.7, color='purple', edgecolor='black', label='预测值')
        plt.hist(y_test.flatten(), bins=20, alpha=0.5, color='cyan', edgecolor='black', label='真实值')
        plt.xlabel('销量 (件)')
        plt.ylabel('频次')
        plt.title('预测值与真实值分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 相对误差分布
        plt.subplot(2, 3, 5)
        relative_errors = np.abs(residuals / y_test.flatten()) * 100
        plt.hist(relative_errors, bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('相对误差 (%)')
        plt.ylabel('频次')
        plt.title('相对误差分布')
        plt.grid(True, alpha=0.3)

        # Q-Q图
        plt.subplot(2, 3, 6)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('残差 Q-Q 图')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片和数据
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图表已保存至: {save_path}")

        # 保存数据到evaluation_data目录
        plot_data = {
            'y_test': y_test.flatten(),
            'y_pred': y_pred,
            'residuals': y_test.flatten() - y_pred,
            'relative_errors': np.abs((y_test.flatten() - y_pred) / y_test.flatten()) * 100
        }
        data_filename = "sales_prediction_results_data.json"
        self.save_plot_data(plot_data, os.path.join("evaluation_data", data_filename))

        plt.close()  # 关闭图形

    def analyze_feature_importance(self, results, X_train, save_path=None):
        """分析特征重要性"""
        plt.figure(figsize=(14, 10))

        best_lambda = self.best_lambda
        weights = results[best_lambda]['weights']

        # 特征权重系数
        plt.subplot(2, 3, 1)
        colors = ['red' if w > 0 else 'blue' for w in weights]

        bars = plt.barh(range(len(weights)), weights, color=colors, alpha=0.7, edgecolor='black')
        plt.yticks(range(len(weights)), self.feature_names)
        plt.xlabel('权重系数')
        plt.ylabel('特征')
        plt.title(f'特征权重系数 (λ={best_lambda})')
        plt.grid(True, alpha=0.3, axis='x')

        # 添加数值标签
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            plt.text(weight + (0.1 if weight > 0 else -0.1), bar.get_y() + bar.get_height()/2,
                    f'{weight:.2f}', ha='left' if weight > 0 else 'right', va='center', fontsize=7)

        # 特征重要性百分比
        plt.subplot(2, 3, 2)
        abs_weights = np.abs(weights)
        importance_percent = (abs_weights / abs_weights.sum()) * 100

        # 创建饼图，不在扇形内显示百分比
        wedges, texts = plt.pie(importance_percent, startangle=90, colors=plt.cm.Set3.colors)
        plt.title('特征重要性百分比')
        plt.axis('equal')

        # 创建图例，显示特征名称和百分比
        legend_labels = [f'{feature}: {percent:.1f}%' for feature, percent in zip(self.feature_names, importance_percent)]
        plt.legend(wedges, legend_labels, title="特征重要性", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

        # 权重绝对值排序
        plt.subplot(2, 3, 3)
        sorted_idx = np.argsort(abs_weights)[::-1]
        sorted_features = [self.feature_names[i] for i in sorted_idx]
        sorted_weights = abs_weights[sorted_idx]

        bars = plt.bar(range(len(sorted_features)), sorted_weights,
                      color='skyblue', alpha=0.7, edgecolor='black')
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
        plt.ylabel('权重绝对值')
        plt.xlabel('特征')
        plt.title('特征重要性排序')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, weight in zip(bars, sorted_weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{weight:.2f}', ha='center', va='bottom', fontsize=8)

        # 不同λ值下各特征的权重变化
        plt.subplot(2, 3, 4)
        lambda_values = list(results.keys())
        weights_matrix = np.array([results[lam]['weights'] for lam in lambda_values])

        for i, feature in enumerate(self.feature_names):
            plt.plot(lambda_values, np.abs(weights_matrix[:, i]), 'o-', label=feature,
                    linewidth=2, markersize=6)

        plt.xscale('log')
        plt.xlabel('正则化参数 λ')
        plt.ylabel('权重绝对值')
        plt.title('各特征权重随λ的变化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # 业务影响分析
        plt.subplot(2, 3, 5)
        # 计算每个特征的边际效应（标准化后的）
        X_mean = np.mean(X_train, axis=0)
        feature_effects = weights * X_mean  # 边际效应

        colors = ['lightgreen' if eff > 0 else 'lightcoral' for eff in feature_effects]
        bars = plt.bar(range(len(feature_effects)), feature_effects,
                      color=colors, alpha=0.7, edgecolor='black')
        plt.xticks(range(len(feature_effects)), self.feature_names, rotation=45, ha='right')
        plt.ylabel('边际效应')
        plt.xlabel('特征')
        plt.title('各特征的边际效应')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, effect in zip(bars, feature_effects):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if effect > 0 else -0.1),
                    f'{effect:.1f}', ha='center', va='bottom' if effect > 0 else 'top', fontsize=7)

        # 特征敏感性分析
        plt.subplot(2, 3, 6)
        # 计算特征的标准差
        X_std = np.std(X_train, axis=0)
        sensitivities = np.abs(weights) * X_std

        sorted_idx = np.argsort(sensitivities)[::-1]
        sorted_features = [self.feature_names[i] for i in sorted_idx]
        sorted_sensitivities = sensitivities[sorted_idx]

        bars = plt.bar(range(len(sorted_features)), sorted_sensitivities,
                      color='lightblue', alpha=0.7, edgecolor='black')
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
        plt.ylabel('敏感性')
        plt.xlabel('特征')
        plt.title('特征敏感性分析')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, sensitivity in zip(bars, sorted_sensitivities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sensitivity:.1f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # 保存图片和数据
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性分析图表已保存至: {save_path}")

        # 保存数据到feature_analysis目录
        plot_data = {
            'feature_names': self.feature_names,
            'weights': weights.tolist(),
            'abs_weights': abs_weights.tolist(),
            'importance_percent': importance_percent.tolist(),
            'sorted_features': sorted_features,
            'sorted_weights': sorted_weights.tolist(),
            'sensitivities': sensitivities.tolist(),
            'X_train_mean': X_train.mean(axis=0).tolist(),
            'X_train_std': X_train.std(axis=0).tolist()
        }
        data_filename = "sales_feature_importance_data.json"
        self.save_plot_data(plot_data, os.path.join("feature_analysis", data_filename))

        plt.close()  # 关闭图形  # 关闭图形而不是显示，避免阻塞

    def business_analysis(self, results):
        """业务分析和运营建议"""
        logger.info("=" * 60)
        logger.info("业务分析和运营建议")
        logger.info("=" * 60)

        best_lambda = self.best_lambda
        weights = results[best_lambda]['weights']

        # 分析各特征的影响
        logger.info("各特征对销量的影响分析:")

        for i, (feature, weight) in enumerate(zip(self.feature_names, weights)):
            if feature == 'price':
                impact = f"价格每上涨1元，销量{('减少' if weight < 0 else '增加')}{abs(weight):.0f}件"
            elif feature == 'promotion':
                impact = f"促销期间销量相比不促销{('增加' if weight > 0 else '减少')}{abs(weight):.0f}件"
            elif feature == 'ad_spend':
                impact = f"广告投入每增加1元，销量{('增加' if weight > 0 else '减少')}{abs(weight):.0f}件"
            elif feature == 'user_rating':
                impact = f"用户评分每提高0.1分，销量{('增加' if weight > 0 else '减少')}{abs(weight):.0f}件"
            elif feature == 'holiday':
                impact = f"节假日销量相比工作日{('增加' if weight > 0 else '减少')}{abs(weight):.0f}件"
            elif feature == 'month':
                impact = f"月份增加1个月，销量{('增加' if weight > 0 else '减少')}{abs(weight):.0f}件"
            elif feature == 'weekday':
                impact = f"星期数增加1天，销量{('增加' if weight > 0 else '减少')}{abs(weight):.0f}件"

            logger.info(f"• {feature}: 权重={weight:.4f}, {impact}")

        # 运营建议
        logger.info("\n运营策略建议:")

        # 促销建议
        if weights[self.feature_names.index('promotion')] > 0:
            logger.info("• 建议增加促销活动频率，促销对销量有显著正向影响")

        # 广告建议
        if weights[self.feature_names.index('ad_spend')] > 0:
            logger.info("• 增加广告投入可以有效提升销量，建议优化广告策略")

        # 价格建议
        price_weight = weights[self.feature_names.index('price')]
        if price_weight < 0:
            logger.info("• 建议适度调整价格策略，当前价格对销量有负向影响")

        # 节假日建议
        if weights[self.feature_names.index('holiday')] > 0:
            logger.info("• 节假日销量更高，建议在节假日加大促销和广告投入")

        # 用户评分建议
        if weights[self.feature_names.index('user_rating')] > 0:
            logger.info("• 提升产品品质和服务可以提高销量，建议关注用户评价")

        logger.info(f"\n最佳正则化参数: λ={best_lambda}")
        logger.info("该参数在防止过拟合和保持模型性能之间取得了最佳平衡")


def main():
    """主函数：运行完整的销量预测系统"""
    logger.info("开始运行电商产品销量预测系统")

    # 初始化预测器
    predictor = ProductSalesPredictor()

    try:
        # 1. 加载数据
        df = predictor.load_data('电商产品销量预测与影响因素分析.txt')

        # 2. 数据预处理
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
            predictor.preprocess_data(df)

        # 3. 重新加载数据用于相关性分析（包含时间特征）
        df_processed = predictor._extract_time_features(df.copy())
        predictor.analyze_correlation(df_processed, save_path=os.path.join(predictor.data_dir, 'plots', 'sales_correlation_analysis.png'))

        # 4. 训练不同λ值的Ridge模型
        lambda_values = [0.01, 0.1, 1, 10, 100]
        results = predictor.train_ridge_models(X_train, y_train, X_val, y_val, lambda_values)

        # 5. λ值对比分析可视化
        predictor.plot_lambda_comparison(results, X_train, X_val, y_train, y_val, save_path=os.path.join(predictor.data_dir, 'plots', 'sales_lambda_comparison.png'))

        # 6. 特征重要性分析
        predictor.analyze_feature_importance(results, X_train, save_path=os.path.join(predictor.data_dir, 'plots', 'sales_feature_importance.png'))

        # 7. 测试集评估
        metrics, y_pred = predictor.evaluate_model(X_test, y_test)

        # 8. 预测结果可视化
        predictor.plot_predictions(y_test, y_pred, save_path=os.path.join(predictor.data_dir, 'plots', 'sales_predictions.png'))

        # 保存训练数据
        training_data = {
            'X_train_shape': X_train.shape,
            'X_val_shape': X_val.shape,
            'X_test_shape': X_test.shape,
            'y_train_shape': y_train.shape,
            'y_val_shape': y_val.shape,
            'y_test_shape': y_test.shape,
            'feature_names': predictor.feature_names,
            'best_lambda': predictor.best_lambda,
            'metrics': metrics,
            'lambda_values_tested': lambda_values
        }
        predictor.save_plot_data(training_data, os.path.join("training_data", "sales_model_training_data.json"))

        # 9. 业务分析和运营建议
        predictor.business_analysis(results)

        # 10. 输出最终结果
        logger.info("=" * 60)
        logger.info("电商产品销量预测系统运行完成")
        logger.info("=" * 60)
        logger.info("模型性能指标:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        raise


if __name__ == "__main__":
    main()
