#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病预测系统（二分类基础）
项目三：基于逻辑回归的糖尿病预测

作者: 王梓涵
邮箱: wangzh011031@163.com
时间: 2025年12月24日

功能：
- 数据标准化与特征可视化
- 手动实现逻辑回归（梯度下降、Sigmoid函数）
- 绘制决策边界与代价函数下降曲线
- 使用sklearn的LogisticRegression验证结果
- 特征重要性与业务分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# 设置科研绘图风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diabetes_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiabetesPredictor:
    """
    糖尿病预测器

    实现逻辑回归模型，包含手动梯度下降实现和sklearn对比。
    """

    def __init__(self, learning_rate=0.1, max_iter=1000, random_state=42):
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
        self.scaler = StandardScaler()
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # 创建数据保存目录
        self.data_dir = "diabetes_data"
        self.create_data_directories()

        logger.info("糖尿病预测器初始化完成")

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
        
        # 如果不是字典，则直接序列化
        if not isinstance(data_dict, dict):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存至: {filepath}")
            return

        serializable_data = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, (int, float, str, list, dict)):
                serializable_data[key] = value
            else:
                serializable_data[key] = str(value)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        logger.info(f"绘图数据已保存至: {filepath}")

    def load_data(self):
        """
        加载糖尿病数据集 (Pima Indians Diabetes)
        """
        try:
            logger.info("尝试从 sklearn.datasets 加载数据集...")
            # Pima Indians Diabetes Dataset 在 OpenML 中的 ID 是 37
            diabetes = fetch_openml(data_id=37, as_frame=True, parser='auto')
            df = diabetes.frame
            
            # 处理目标变量，确保是 0/1
            if df['class'].dtype == 'category' or df['class'].dtype == 'object':
                # 假设 'tested_positive' 为 1, 'tested_negative' 为 0
                df['Outcome'] = df['class'].apply(lambda x: 1 if 'pos' in str(x).lower() else 0)
                df = df.drop(columns=['class'])
            else:
                df = df.rename(columns={'class': 'Outcome'})
            
            # 重命名列以匹配项目描述
            column_mapping = {
                'preg': 'Pregnancies',
                'plas': 'Glucose',
                'pres': 'BloodPressure',
                'skin': 'SkinThickness',
                'insu': 'Insulin',
                'mass': 'BMI',
                'pedi': 'DiabetesPedigreeFunction',
                'age': 'Age'
            }
            df = df.rename(columns=column_mapping)
            
            logger.info(f"成功加载数据，样本数量: {len(df)}")
            return df
        except Exception as e:
            logger.warning(f"从 OpenML 加载失败: {e}. 使用模拟数据进行项目演示。")
            # 模拟数据生成
            np.random.seed(self.random_state)
            n_samples = 768
            data = {
                'Pregnancies': np.random.randint(0, 18, n_samples),
                'Glucose': np.random.randint(44, 199, n_samples),
                'BloodPressure': np.random.randint(24, 122, n_samples),
                'SkinThickness': np.random.randint(7, 99, n_samples),
                'Insulin': np.random.randint(14, 846, n_samples),
                'BMI': np.random.uniform(18, 67, n_samples),
                'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.42, n_samples),
                'Age': np.random.randint(21, 81, n_samples),
            }
            df = pd.DataFrame(data)
            # 简单逻辑生成 Outcome
            logits = (df['Glucose'] * 0.04 + df['BMI'] * 0.08 + df['Age'] * 0.02 - 10)
            probs = 1 / (1 + np.exp(-logits))
            df['Outcome'] = (probs > np.random.random(n_samples)).astype(int)
            
            logger.info(f"生成模拟数据，样本数量: {len(df)}")
            return df

    def preprocess_data(self, df):
        """数据预处理：标准化"""
        logger.info("开始数据预处理...")
        
        # 确保转换为数值型
        X = df[self.feature_names].to_numpy().astype(float)
        y = df['Outcome'].to_numpy().astype(float).reshape(-1, 1)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    def sigmoid(self, z):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, weights, bias):
        """计算逻辑回归代价函数 (交叉熵损失)"""
        m = len(y)
        z = np.dot(X, weights) + bias
        h = self.sigmoid(z)
        # 防止 log(0)
        epsilon = 1e-15
        cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost

    def manual_gradient_descent(self, X, y):
        """手动实现梯度下降"""
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0.0
        
        costs = []
        iterations = []

        for i in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)
            
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                cost = self.compute_cost(X, y, self.weights, self.bias)
                costs.append(cost)
                iterations.append(i)
                if i % 500 == 0:
                    logger.info(f"迭代 {i}, 代价: {cost:.6f}")

        logger.info(f"手动梯度下降训练完成，最终代价: {costs[-1]:.6f}")
        return {'costs': costs, 'iterations': iterations}

    def predict_proba(self, X_scaled):
        """预测概率"""
        z = np.dot(X_scaled, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X_scaled, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X_scaled)
        return (proba >= threshold).astype(int)

    def plot_data_analysis(self, df, save_path=None):
        """特征分布与相关性可视化"""
        plt.figure(figsize=(16, 12))
        
        # 1. 目标变量分布
        plt.subplot(3, 3, 1)
        sns.countplot(x='Outcome', data=df, palette='viridis')
        plt.title('糖尿病患病分布 (0:正常, 1:患病)')
        
        # 2. 血糖 vs BMI 散点图
        plt.subplot(3, 3, 2)
        sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, alpha=0.6)
        plt.title('血糖 vs BMI 相关性')
        
        # 3. 年龄分布
        plt.subplot(3, 3, 3)
        sns.histplot(df['Age'], kde=True, color='orange')
        plt.title('年龄分布')
        
        # 4. 特征箱线图
        plt.subplot(3, 3, 4)
        sns.boxplot(x='Outcome', y='Glucose', data=df)
        plt.title('不同分类的血糖水平对比')
        
        # 5. 相关性热力图
        plt.subplot(3, 3, 5)
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
        plt.title('特征相关性热力图')
        
        # 6-9. 其他重要特征分布
        features_to_plot = ['Pregnancies', 'BloodPressure', 'Insulin', 'DiabetesPedigreeFunction']
        for i, feat in enumerate(features_to_plot):
            plt.subplot(3, 3, 6 + i)
            sns.kdeplot(data=df, x=feat, hue='Outcome', fill=True)
            plt.title(f'{feat} 分布对比')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"数据分析图保存至: {save_path}")
        
        # 详细分布数据保存
        distribution_data = {
            'total_count': int(len(df)),
            'outcome_counts': df['Outcome'].value_counts().to_dict(),
            'outcome_ratio': (df['Outcome'].value_counts(normalize=True)).to_dict(),
            'age_groups': {
                '20-30': df[(df['Age'] >= 20) & (df['Age'] < 30)]['Outcome'].value_counts(normalize=True).to_dict(),
                '30-40': df[(df['Age'] >= 30) & (df['Age'] < 40)]['Outcome'].value_counts(normalize=True).to_dict(),
                '40-50': df[(df['Age'] >= 40) & (df['Age'] < 50)]['Outcome'].value_counts(normalize=True).to_dict(),
                '50+': df[df['Age'] >= 50]['Outcome'].value_counts(normalize=True).to_dict()
            },
            'bmi_groups': {
                'underweight_normal': df[df['BMI'] < 25]['Outcome'].value_counts(normalize=True).to_dict(),
                'overweight': df[(df['BMI'] >= 25) & (df['BMI'] < 30)]['Outcome'].value_counts(normalize=True).to_dict(),
                'obese': df[df['BMI'] >= 30]['Outcome'].value_counts(normalize=True).to_dict()
            },
            'glucose_groups': {
                'normal': df[df['Glucose'] < 100]['Outcome'].value_counts(normalize=True).to_dict(),
                'prediabetes': df[(df['Glucose'] >= 100) & (df['Glucose'] < 126)]['Outcome'].value_counts(normalize=True).to_dict(),
                'diabetes_range': df[df['Glucose'] >= 126]['Outcome'].value_counts(normalize=True).to_dict()
            }
        }
        
        # 保存详细分布数据
        self.save_plot_data(distribution_data, "plots/detailed_distribution.json")
        # 保存相关性数据
        self.save_plot_data({'correlation': corr.to_dict()}, "plots/correlation_data.json")
        plt.close()

    def plot_training_history(self, history, save_path=None):
        """绘制代价函数下降曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['iterations'], history['costs'], marker='o', linestyle='-', color='b')
        plt.title('代价函数下降曲线 (Manual Logistic Regression)')
        plt.xlabel('迭代次数')
        plt.ylabel('代价 (Cost)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"训练历史图保存至: {save_path}")
        
        self.save_plot_data(history, "plots/training_history_data.json")
        plt.close()

    def plot_evaluation(self, y_true, y_pred, y_prob, save_path=None):
        """评估可视化：混淆矩阵与 ROC 曲线"""
        plt.figure(figsize=(12, 5))
        
        # 混淆矩阵
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        
        # ROC 曲线
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"评估结果图保存至: {save_path}")
        
        eval_data = {
            'confusion_matrix': cm.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        }
        self.save_plot_data(eval_data, "evaluation_data/evaluation_results.json")
        plt.close()

    def analyze_features(self, save_path=None):
        """分析特征权重"""
        weights = self.weights.flatten()
        # 逻辑回归中，权重反映了特征对 log-odds 的贡献
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Weight': weights,
            'AbsWeight': np.abs(weights)
        }).sort_values(by='AbsWeight', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Weight', y='Feature', data=importance, palette='coolwarm')
        plt.title('特征系数分析 (对患病概率的影响程度)')
        plt.grid(True, axis='x')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"特征分析图保存至: {save_path}")
            
        self.save_plot_data(importance.to_dict(orient='records'), "feature_analysis/feature_importance.json")
        plt.close()
        return importance

    def plot_decision_boundary(self, X_scaled, y, save_path=None):
        """绘制决策边界 (选取最重要的两个特征：通常是 Glucose 和 BMI)"""
        # 找到最重要的两个特征索引
        importance = self.analyze_features()
        top_features = importance['Feature'].iloc[:2].values
        idx1 = self.feature_names.index(top_features[0])
        idx2 = self.feature_names.index(top_features[1])
        
        # 简化版：仅使用这两个特征重新训练一个简单的模型用于绘图
        X_sub = X_scaled[:, [idx1, idx2]]
        
        # 绘制网格
        h = .02
        x_min, x_max = X_sub[:, 0].min() - 0.5, X_sub[:, 0].max() + 0.5
        y_min, y_max = X_sub[:, 1].min() - 0.5, X_sub[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # 使用全局模型的权重计算这些点的概率（其他特征取均值0）
        grid_points = np.zeros((xx.ravel().shape[0], len(self.feature_names)))
        grid_points[:, idx1] = xx.ravel()
        grid_points[:, idx2] = yy.ravel()
        
        Z = self.predict(grid_points).reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.scatter(X_sub[:, 0], X_sub[:, 1], c=y.flatten(), edgecolors='k', cmap='RdYlBu', alpha=0.7)
        plt.xlabel(f'标准化 {top_features[0]}')
        plt.ylabel(f'标准化 {top_features[1]}')
        plt.title(f'逻辑回归决策边界 ({top_features[0]} vs {top_features[1]})')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"决策边界图保存至: {save_path}")
        plt.close()

def main():
    logger.info("开始运行糖尿病预测系统...")
    
    predictor = DiabetesPredictor(learning_rate=0.5, max_iter=2000)
    
    # 1. 加载数据
    df = predictor.load_data()
    
    # 2. 数据分析
    predictor.plot_data_analysis(df, save_path=os.path.join(predictor.data_dir, "plots/data_analysis.png"))
    
    # 3. 预处理
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = predictor.preprocess_data(df)
    
    # 4. 手动训练
    history = predictor.manual_gradient_descent(X_train_scaled, y_train)
    predictor.plot_training_history(history, save_path=os.path.join(predictor.data_dir, "plots/training_history.png"))
    
    # 5. 模型评估
    y_prob = predictor.predict_proba(X_test_scaled)
    y_pred = predictor.predict(X_test_scaled)
    
    predictor.plot_evaluation(y_test, y_pred, y_prob, save_path=os.path.join(predictor.data_dir, "plots/model_evaluation.png"))
    
    # 6. 特征分析
    importance = predictor.analyze_features(save_path=os.path.join(predictor.data_dir, "plots/feature_importance.png"))
    
    # 7. 决策边界
    predictor.plot_decision_boundary(X_train_scaled, y_train, save_path=os.path.join(predictor.data_dir, "plots/decision_boundary.png"))
    
    # 8. Sklearn 验证
    lr_sklearn = LogisticRegression(random_state=42)
    lr_sklearn.fit(X_train_scaled, y_train.ravel())
    y_pred_sklearn = lr_sklearn.predict(X_test_scaled)
    
    acc_manual = accuracy_score(y_test, y_pred)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    logger.info("=" * 50)
    logger.info(f"手动实现准确率: {acc_manual:.4f}")
    logger.info(f"Sklearn 实现准确率: {acc_sklearn:.4f}")
    logger.info("=" * 50)
    
    # 9. 业务结论
    logger.info("特征影响分析:")
    for _, row in importance.iterrows():
        direction = "正向（增加患病风险）" if row['Weight'] > 0 else "负向（降低患病风险）"
        logger.info(f"• {row['Feature']}: 权重 {row['Weight']:.4f}, 影响: {direction}")

    logger.info("糖尿病预测系统运行完成。")

if __name__ == "__main__":
    main()

