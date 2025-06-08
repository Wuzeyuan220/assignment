"""
preprocessing.py - 数据预处理模块

该模块负责对原始数据进行清洗、转换和特征工程，为后续的分析和建模做准备。
包含数据清洗、缺失值处理、异常值检测、特征编码等功能。
"""

import pandas as pd
import numpy as np
import re
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器类，负责数据清洗和特征工程"""
    
    def __init__(self):
        """初始化数据预处理器"""
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.transformers = {}
    
    def clean_data(self, df):
        """
        执行基本的数据清洗
        
        参数:
            df: 原始数据DataFrame
            
        返回:
            pandas.DataFrame: 清洗后的数据
        """
        logger.info("开始数据清洗...")
        
        # 复制数据，避免修改原始数据
        cleaned_df = df.copy()
        
        # 检查并处理重复值
        duplicates = cleaned_df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"检测到 {duplicates} 个重复行，已删除")
            cleaned_df = cleaned_df.drop_duplicates()
        
        # 标准化列名（转为小写并替换空格为下划线）
        cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
        
        # 检查并处理列名中的特殊字符
        invalid_chars = r'[^a-zA-Z0-9_]'
        cleaned_df.columns = [re.sub(invalid_chars, '', col) for col in cleaned_df.columns]
        
        logger.info("数据清洗完成")
        return cleaned_df
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        处理缺失值
        
        参数:
            df: 数据DataFrame
            strategy: 处理策略 ('auto', 'drop', 'mean', 'median', 'mode', 'ffill', 'bfill')
            
        返回:
            pandas.DataFrame: 处理后的数据
        """
        logger.info("开始处理缺失值...")
        
        # 复制数据，避免修改原始数据
        processed_df = df.copy()
        
        # 检查缺失值
        missing_counts = processed_df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) == 0:
            logger.info("未检测到缺失值")
            return processed_df
        
        logger.info(f"检测到缺失值的列:\n{missing_cols}")
        
        # 根据策略处理缺失值
        if strategy == 'auto':
            # 自动选择策略：数值列用中位数填充，分类列用众数填充
            for col in missing_cols.index:
                if processed_df[col].dtype in [np.number, 'int64', 'float64']:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                else:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
        elif strategy == 'drop':
            # 删除包含缺失值的行
            processed_df = processed_df.dropna()
        elif strategy in ['mean', 'median', 'mode', 'ffill', 'bfill']:
            # 使用指定策略填充
            for col in missing_cols.index:
                if processed_df[col].dtype in [np.number, 'int64', 'float64']:
                    if strategy == 'mean':
                        processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        processed_df[col].fillna(processed_df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                    elif strategy == 'ffill':
                        processed_df[col].fillna(method='ffill', inplace=True)
                    elif strategy == 'bfill':
                        processed_df[col].fillna(method='bfill', inplace=True)
                else:
                    if strategy in ['mode', 'ffill', 'bfill']:
                        if strategy == 'mode':
                            processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                        elif strategy == 'ffill':
                            processed_df[col].fillna(method='ffill', inplace=True)
                        elif strategy == 'bfill':
                            processed_df[col].fillna(method='bfill', inplace=True)
                    else:
                        logger.warning(f"策略 {strategy} 不适用于非数值列 {col}，使用众数填充")
                        processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
        
        # 检查处理后的缺失值
        remaining_missing = processed_df.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"仍有 {remaining_missing} 个缺失值未处理")
        
        logger.info("缺失值处理完成")
        return processed_df
    
    def detect_and_handle_outliers(self, df, columns=None, method='iqr', threshold=1.5, action='clip'):
        """
        检测并处理异常值
        
        参数:
            df: 数据DataFrame
            columns: 需要处理的列名列表，默认为所有数值列
            method: 异常值检测方法 ('iqr', 'zscore')
            threshold: 阈值参数
            action: 处理动作 ('remove', 'clip', 'flag')
            
        返回:
            pandas.DataFrame: 处理后的数据
        """
        logger.info("开始检测和处理异常值...")
        
        # 复制数据，避免修改原始数据
        processed_df = df.copy()
        
        # 如果未指定列，则选择所有数值列
        if columns is None:
            columns = processed_df.select_dtypes(include=[np.number, 'int64', 'float64']).columns.tolist()
        
        if not columns:
            logger.info("没有找到数值列进行异常值处理")
            return processed_df
        
        logger.info(f"将处理以下列的异常值: {columns}")
        
        for col in columns:
            if method == 'iqr':
                # 使用四分位距法检测异常值
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"列 {col} 检测到 {outlier_count} 个异常值")
                    
                    if action == 'remove':
                        # 删除包含异常值的行
                        processed_df = processed_df[~outliers]
                    elif action == 'clip':
                        # 截断异常值（将异常值替换为上下限）
                        processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
                    elif action == 'flag':
                        # 添加异常值标记列
                        processed_df[f"{col}_is_outlier"] = outliers
            
            elif method == 'zscore':
                # 使用Z-score法检测异常值
                mean = processed_df[col].mean()
                std = processed_df[col].std()
                
                # 处理标准差为0的情况
                if std == 0:
                    logger.warning(f"列 {col} 的标准差为0，跳过异常值检测")
                    continue
                
                z_scores = (processed_df[col] - mean) / std
                outliers = (abs(z_scores) > threshold)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"列 {col} 检测到 {outlier_count} 个异常值")
                    
                    if action == 'remove':
                        # 删除包含异常值的行
                        processed_df = processed_df[~outliers]
                    elif action == 'clip':
                        # 截断异常值
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
                    elif action == 'flag':
                        # 添加异常值标记列
                        processed_df[f"{col}_is_outlier"] = outliers
        
        logger.info("异常值处理完成")
        return processed_df
    
    def encode_categorical_features(self, df, columns=None, method='onehot'):
        """
        对分类特征进行编码
        
        参数:
            df: 数据DataFrame
            columns: 需要编码的列名列表，默认为所有分类列
            method: 编码方法 ('onehot', 'ordinal', 'label')
            
        返回:
            pandas.DataFrame: 编码后的数据
        """
        logger.info("开始对分类特征进行编码...")
        
        # 复制数据，避免修改原始数据
        encoded_df = df.copy()
        
        # 如果未指定列，则选择所有分类列
        if columns is None:
            columns = encoded_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not columns:
            logger.info("没有找到分类列进行编码")
            return encoded_df
        
        logger.info(f"将对以下列进行编码: {columns}")
        
        if method == 'onehot':
            # 使用独热编码
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(encoded_df[columns])
            
            # 创建编码后的列名
            feature_names = encoder.get_feature_names_out(columns)
            
            # 删除原始列
            encoded_df = encoded_df.drop(columns=columns)
            
            # 添加编码后的列
            encoded_df = pd.concat([
                encoded_df.reset_index(drop=True),
                pd.DataFrame(encoded_features, columns=feature_names)
            ], axis=1)
            
            # 保存编码器以便后续使用
            self.transformers['onehot_encoder'] = encoder
            
        elif method == 'ordinal':
            # 使用序数编码
            for col in columns:
                # 获取唯一值并排序
                unique_values = sorted(encoded_df[col].dropna().unique())
                
                # 创建映射字典
                value_map = {value: i for i, value in enumerate(unique_values)}
                
                # 应用映射
                encoded_df[col] = encoded_df[col].map(value_map)
                
                # 处理缺失值
                encoded_df[col] = encoded_df[col].fillna(-1)  # 使用-1表示缺失值
                
                # 保存映射以便后续使用
                if 'ordinal_encoders' not in self.transformers:
                    self.transformers['ordinal_encoders'] = {}
                self.transformers['ordinal_encoders'][col] = value_map
        
        elif method == 'label':
            # 使用标签编码
            for col in columns:
                # 获取唯一值并排序
                unique_values = sorted(encoded_df[col].dropna().unique())
                
                # 创建映射字典
                value_map = {value: i for i, value in enumerate(unique_values)}
                
                # 应用映射
                encoded_df[col] = encoded_df[col].map(value_map)
                
                # 处理缺失值
                encoded_df[col] = encoded_df[col].fillna(-1)  # 使用-1表示缺失值
                
                # 保存映射以便后续使用
                if 'label_encoders' not in self.transformers:
                    self.transformers['label_encoders'] = {}
                self.transformers['label_encoders'][col] = value_map
        
        logger.info("分类特征编码完成")
        return encoded_df
    
    def normalize_numeric_features(self, df, columns=None, method='standard'):
        """
        对数值特征进行归一化
        
        参数:
            df: 数据DataFrame
            columns: 需要归一化的列名列表，默认为所有数值列
            method: 归一化方法 ('standard', 'minmax', 'log')
            
        返回:
            pandas.DataFrame: 归一化后的数据
        """
        logger.info("开始对数值特征进行归一化...")
        
        # 复制数据，避免修改原始数据
        normalized_df = df.copy()
        
        # 如果未指定列，则选择所有数值列
        if columns is None:
            columns = normalized_df.select_dtypes(include=[np.number, 'int64', 'float64']).columns.tolist()
        
        if not columns:
            logger.info("没有找到数值列进行归一化")
            return normalized_df
        
        logger.info(f"将对以下列进行归一化: {columns}")
        
        if method == 'standard':
            # 使用标准缩放（Z-score标准化）
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(normalized_df[columns])
            
            # 创建归一化后的DataFrame
            normalized_df[columns] = normalized_features
            
            # 保存缩放器以便后续使用
            self.transformers['standard_scaler'] = scaler
            
        elif method == 'minmax':
            # 使用最小-最大缩放
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            normalized_features = scaler.fit_transform(normalized_df[columns])
            
            # 创建归一化后的DataFrame
            normalized_df[columns] = normalized_features
            
            # 保存缩放器以便后续使用
            self.transformers['minmax_scaler'] = scaler
            
        elif method == 'log':
            # 使用对数变换
            for col in columns:
                # 处理非正值
                min_val = normalized_df[col].min()
                if min_val <= 0:
                    # 添加一个小的偏移量使所有值为正
                    offset = abs(min_val) + 1
                    normalized_df[col] = normalized_df[col] + offset
                
                # 应用对数变换
                normalized_df[col] = np.log(normalized_df[col])
        
        logger.info("数值特征归一化完成")
        return normalized_df
    
    def process_text(self, text):
        """
        处理文本数据（分词、去停用词、词形还原）
        
        参数:
            text: 输入文本
            
        返回:
            str: 处理后的文本
        """
        if pd.isna(text):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 分词
        tokens = nltk.word_tokenize(text)
        
        # 移除停用词并进行词形还原
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        
        # 重新组合为文本
        return " ".join(processed_tokens)
    
    def create_text_features(self, df, text_column, max_features=1000):
        """
        从文本列创建特征
        
        参数:
            df: 数据DataFrame
            text_column: 文本列名
            max_features: 最大特征数
            
        返回:
            pandas.DataFrame: 添加了文本特征的数据
        """
        logger.info(f"从文本列 {text_column} 创建特征...")
        
        # 复制数据，避免修改原始数据
        feature_df = df.copy()
        
        # 处理文本
        feature_df[f"{text_column}_processed"] = feature_df[text_column].apply(self.process_text)
        
        # 创建TF-IDF特征
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(feature_df[f"{text_column}_processed"])
        
        # 创建特征列名
        feature_names = [f"{text_column}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        
        # 将TF-IDF矩阵转换为DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        
        # 合并特征
        feature_df = pd.concat([
            feature_df.reset_index(drop=True),
            tfidf_df
        ], axis=1)
        
        # 保存向量化器以便后续使用
        self.transformers['tfidf_vectorizer'] = vectorizer
        
        logger.info(f"从文本列 {text_column} 创建了 {len(feature_names)} 个特征")
        return feature_df
    
    def create_time_features(self, df, date_column):
        """
        从日期列创建时间特征
        
        参数:
            df: 数据DataFrame
            date_column: 日期列名
            
        返回:
            pandas.DataFrame: 添加了时间特征的数据
        """
        logger.info(f"从日期列 {date_column} 创建时间特征...")
        
        # 复制数据，避免修改原始数据
        time_df = df.copy()
        
        # 确保日期列是datetime类型
        time_df[date_column] = pd.to_datetime(time_df[date_column])
        
        # 创建时间特征
        time_df[f"{date_column}_year"] = time_df[date_column].dt.year
        time_df[f"{date_column}_month"] = time_df[date_column].dt.month
        time_df[f"{date_column}_day"] = time_df[date_column].dt.day
        time_df[f"{date_column}_dayofweek"] = time_df[date_column].dt.dayofweek
        time_df[f"{date_column}_hour"] = time_df[date_column].dt.hour
        time_df[f"{date_column}_is_weekend"] = time_df[date_column].dt.dayofweek >= 5
        
        logger.info(f"从日期列 {date_column} 创建了时间特征")
        return time_df
    
    def create_pipeline(self, numeric_features, categorical_features, text_features=None, date_features=None):
        """
        创建数据预处理管道
        
        参数:
            numeric_features: 数值特征列表
            categorical_features: 分类特征列表
            text_features: 文本特征字典 {列名: 最大特征数}
            date_features: 日期特征列表
            
        返回:
            sklearn.pipeline.Pipeline: 预处理管道
        """
        logger.info("创建数据预处理管道...")
        
        # 创建数值特征处理管道
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 创建分类特征处理管道
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # 组合所有特征处理步骤
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # 创建完整的预处理管道
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        # 保存管道以便后续使用
        self.transformers['preprocessing_pipeline'] = pipeline
        
        logger.info("数据预处理管道创建完成")
        return pipeline

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, None, 40, 35],
        'gender': ['F', 'M', 'M', 'M', 'F'],
        'income': [50000, 60000, 75000, 90000, 80000],
        'text': [
            'This is a sample text with some words.',
            'Another example text for demonstration.',
            'Data preprocessing is important for machine learning.',
            'Feature engineering can improve model performance.',
            'Natural language processing involves text analysis.'
        ],
        'date': ['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-30', '2023-05-10']
    }
    
    df = pd.DataFrame(data)
    
    # 创建数据预处理器实例
    preprocessor = DataPreprocessor()
    
    # 数据清洗
    cleaned_df = preprocessor.clean_data(df)
    
    # 处理缺失值
    filled_df = preprocessor.handle_missing_values(cleaned_df)
    
    # 检测并处理异常值
    processed_df = preprocessor.detect_and_handle_outliers(filled_df, columns=['income'])
    
    # 编码分类特征
    encoded_df = preprocessor.encode_categorical_features(processed_df, columns=['gender'])
    
    # 归一化数值特征
    normalized_df = preprocessor.normalize_numeric_features(encoded_df, columns=['age', 'income'])
    
    # 创建文本特征
    text_featured_df = preprocessor.create_text_features(normalized_df, 'text', max_features=5)
    
    # 创建时间特征
    final_df = preprocessor.create_time_features(text_featured_df, 'date')
    
    print("预处理完成的数据:")
    print(final_df.head())
