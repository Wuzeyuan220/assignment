# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.models = {}
        self.imputers = {}  # 为不同模型类型存储独立的imputer
        self.feature_selectors = {}  # 存储特征选择器
        
    def perform_eda(self, data):
        """执行探索性数据分析"""
        logger.info("开始探索性数据分析...")
        results = {
            'shape': data.shape,
            'columns': list(data.columns),
            'data_types': {col: data[col].dtype for col in data.columns},
            'missing_values': {col: data[col].isnull().sum() for col in data.columns},
            'summary_stats': data.describe().to_dict()
        }
        logger.info("探索性数据分析完成")
        return results
    
    def visualize_distributions(self, data):
        """可视化数值特征分布"""
        logger.info("可视化数值特征分布...")
        fig, axes = plt.subplots(len(data.columns)//3 + 1, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(data.columns):
            if i < len(axes):
                sns.histplot(data[col], kde=True, ax=axes[i])
                axes[i].set_title(f'{col} 分布')
        
        plt.tight_layout()
        logger.info("数值特征分布可视化完成")
        return fig
    
    def visualize_correlations(self, data):
        """可视化特征相关性"""
        logger.info("可视化特征相关性...")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('特征相关性热力图')
        plt.tight_layout()
        logger.info("特征相关性可视化完成")
        return fig
    
    def select_features(self, X, y, method='f_test', k=5, model_key=None):
        """
        特征选择
        
        参数:
            X: 特征
            y: 标签
            method: 特征选择方法
            k: 选择的特征数量
            model_key: 模型键，用于存储特征选择器
            
        返回:
            tuple: (选择的特征, 选择的特征索引, 特征分数, 选择的特征名称)
        """
        logger.info(f"执行特征选择，方法: {method}, 选择 {k} 个特征...")
        
        if method == 'f_test':
            # 根据问题类型选择适当的测试
            if len(np.unique(y)) <= 10:  # 分类问题
                selector = SelectKBest(f_classif, k=k)
            else:  # 回归问题
                selector = SelectKBest(f_regression, k=k)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        X_selected = selector.fit_transform(X, y)
        indices = selector.get_support(indices=True)
        scores = selector.scores_
        selected_features = X.columns[indices].tolist()
        
        # 保存特征选择器
        if model_key:
            self.feature_selectors[model_key] = {
                'selector': selector,
                'indices': indices,
                'selected_features': selected_features
            }
        
        logger.info(f"特征选择完成，选择了 {k} 个特征")
        return X_selected, indices, scores, selected_features
    
    def train_regression_model(self, X, y, model_type='random_forest'):
        """
        训练回归模型
        
        参数:
            X: 特征
            y: 标签
            model_type: 模型类型
            
        返回:
            dict: 模型结果
        """
        logger.info(f"训练回归模型，类型: {model_type}...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 创建并保存imputer
        model_key = f'{model_type}_regression'
        self.imputers[model_key] = SimpleImputer(strategy='mean')
        
        # 处理缺失值
        logger.info("处理缺失值...")
        X_train_imputed = self.imputers[model_key].fit_transform(X_train)
        X_test_imputed = self.imputers[model_key].transform(X_test)
        
        # 特征选择
        logger.info("执行特征选择...")
        X_train_selected, indices, scores, selected_features = self.select_features(
            pd.DataFrame(X_train_imputed, columns=X_train.columns), 
            y_train, 
            k=5,
            model_key=model_key
        )
        
        # 创建模型
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        model.fit(X_train_selected, y_train)
        
        # 保存模型
        self.models[model_key] = model
        
        # 评估模型
        X_test_selected = self.transform_with_selected_features(
            pd.DataFrame(X_test_imputed, columns=X_test.columns), 
            model_key
        )
        y_pred = model.predict(X_test_selected)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"回归模型训练完成，测试集RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return {
            'model': model,
            'test_rmse': rmse,
            'r2': r2,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'selected_features': selected_features
        }
    
    def train_classification_model(self, X, y, model_type='random_forest'):
        """
        训练分类模型
        
        参数:
            X: 特征
            y: 标签
            model_type: 模型类型
            
        返回:
            dict: 模型结果
        """
        logger.info(f"训练分类模型，类型: {model_type}...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 创建并保存imputer
        model_key = f'{model_type}_classification'
        self.imputers[model_key] = SimpleImputer(strategy='mean')
        
        # 处理缺失值
        logger.info("处理缺失值...")
        X_train_imputed = self.imputers[model_key].fit_transform(X_train)
        X_test_imputed = self.imputers[model_key].transform(X_test)
        
        # 特征选择
        logger.info("执行特征选择...")
        X_train_selected, indices, scores, selected_features = self.select_features(
            pd.DataFrame(X_train_imputed, columns=X_train.columns), 
            y_train, 
            k=5,
            model_key=model_key
        )
        
        # 创建模型
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        model.fit(X_train_selected, y_train)
        
        # 保存模型
        self.models[model_key] = model
        
        # 评估模型
        X_test_selected = self.transform_with_selected_features(
            pd.DataFrame(X_test_imputed, columns=X_test.columns), 
            model_key
        )
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"分类模型训练完成，测试集准确率: {accuracy:.4f}")
        return {
            'model': model,
            'test_accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'selected_features': selected_features
        }
    
    def transform_with_selected_features(self, X, model_key):
        """使用保存的特征选择器转换数据"""
        if model_key not in self.feature_selectors:
            return X
        
        selector_info = self.feature_selectors[model_key]
        indices = selector_info['indices']
        selected_features = selector_info['selected_features']
        
        # 确保列名匹配
        if isinstance(X, pd.DataFrame):
            X_selected = X.iloc[:, indices]
            X_selected.columns = selected_features
            return X_selected
        else:
            return X[:, indices]
        
    def calculate_correlation(self, data):
        """
        计算特征之间的相关系数矩阵

        参数:
            data: 数据DataFrame

        返回:
            pandas.DataFrame: 相关系数矩阵
        """
        logger.info("计算特征相关系数矩阵...")

        # 确保数据是数值型
        numeric_data = data.select_dtypes(include=[np.number, 'int64', 'float64'])

        # 计算相关系数矩阵
        correlation_matrix = numeric_data.corr()

        logger.info("特征相关系数矩阵计算完成")
        return correlation_matrix
    
    def descriptive_statistics(self, data):
        """计算数据的描述性统计信息"""
        if isinstance(data, pd.DataFrame):
            return data.describe()
        else:
            # 如果数据不是DataFrame，转换为DataFrame再计算
            df = pd.DataFrame(data)
            return df.describe()
    
    
    def visualize_model_results(self, model_key, X_test, y_test):
        """
        可视化模型结果
        
        参数:
            model_key: 模型键
            X_test: 测试特征
            y_test: 测试标签
            
        返回:
            matplotlib.figure.Figure: 生成的图形
        """
        logger.info(f"可视化模型 {model_key} 结果...")
        
        # 检查模型是否存在
        if model_key not in self.models:
            logger.error(f"模型 {model_key} 不存在")
            return None
        
        model = self.models[model_key]
        
        # 检查imputer是否存在
        if model_key not in self.imputers:
            logger.error(f"imputer for {model_key} 不存在")
            return None
        
        # 处理缺失值
        logger.info("处理缺失值...")
        
        # 确保使用正确的imputer
        X_test_imputed = self.imputers[model_key].transform(X_test)
        
        # 如果是DataFrame，使用原始列名
        if isinstance(X_test, pd.DataFrame):
            X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)
        else:
            # 如果是numpy数组，创建默认列名
            X_test_imputed = pd.DataFrame(X_test_imputed, 
                                         columns=[f"feature_{i}" for i in range(X_test_imputed.shape[1])])
        
        # 应用特征选择
        X_test_selected = self.transform_with_selected_features(X_test_imputed, model_key)
        
        # 创建图形
        fig = plt.figure(figsize=(12, 8))
        
        # 预测
        y_pred = model.predict(X_test_selected)
        
        # 绘制实际值与预测值的对比
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(y_test, y_pred, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax1.set_xlabel('实际值')
        ax1.set_ylabel('预测值')
        ax1.set_title('实际值 vs 预测值')
        
        # 绘制残差图
        ax2 = fig.add_subplot(2, 2, 2)
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('预测值')
        ax2.set_ylabel('残差')
        ax2.set_title('残差图')
        
        # 绘制残差分布
        ax3 = fig.add_subplot(2, 2, 3)
        sns.histplot(residuals, kde=True, ax=ax3)
        ax3.set_xlabel('残差')
        ax3.set_ylabel('频率')
        ax3.set_title('残差分布')
        
        # 绘制特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            ax4 = fig.add_subplot(2, 2, 4)
            feature_importances = pd.DataFrame({
                'feature': X_test_selected.columns,
                'importance': model.feature_importances_
            })
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            sns.barplot(x='importance', y='feature', data=feature_importances.head(10), ax=ax4)
            ax4.set_title('特征重要性')
        
        plt.tight_layout()
        return fig

# 主程序
if __name__ == "__main__":
    # 导入数据集
    logger.info("加载糖尿病数据集...")
    diabetes = load_diabetes()
    X_reg = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y_reg = pd.Series(diabetes.target, name='disease_progression')
    
    logger.info("加载乳腺癌数据集...")
    cancer = load_breast_cancer()
    X_class = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_class = pd.Series(cancer.target, name='target')
    
    # 创建数据分析器
    analyzer = DataAnalyzer()
    
    # 执行探索性分析
    eda_results = analyzer.perform_eda(X_reg.join(y_reg))
    print("探索性分析结果:")
    print(f"shape: {eda_results['shape']}")
    print(f"columns: {eda_results['columns']}")
    print(f"data_types: {eda_results['data_types']}")
    print(f"missing_values: {eda_results['missing_values']}")
    print(f"summary_stats: {eda_results['summary_stats']}")
    
    # 可视化数值特征分布
    analyzer.visualize_distributions(X_reg)
    
    # 可视化特征相关性
    analyzer.visualize_correlations(X_reg.join(y_reg))
    
    # 训练回归模型
    reg_results = analyzer.train_regression_model(X_reg, y_reg)
    print(f"回归模型结果: RMSE={reg_results['test_rmse']:.4f}, R²={reg_results['r2']:.4f}")
    
    # 训练分类模型
    class_results = analyzer.train_classification_model(X_class, y_class)
    print(f"分类模型结果: 准确率={class_results['test_accuracy']:.4f}")
    
    # 可视化回归模型结果
    model_fig = analyzer.visualize_model_results('random_forest_regression', X_reg, y_reg)
    
    logger.info("分析完成！")