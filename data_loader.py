"""
data_loader.py - 数据加载与获取模块

该模块负责从各种来源获取数据，并进行初步的数据加载和解析。
支持本地文件加载、API调用和网页爬虫等多种数据获取方式。
"""

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
# from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
# load_dotenv()

class DataLoader:
    """数据加载器类，负责从不同来源获取数据"""
    
    def __init__(self, data_dir="data"):
        """
        初始化数据加载器
        
        参数:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_from_csv(self, file_path, **kwargs):
        """
        从CSV文件加载数据
        
        参数:
            file_path: CSV文件路径
            **kwargs: pandas.read_csv()的其他参数
            
        返回:
            pandas.DataFrame: 加载的数据
        """
        try:
            logger.info(f"从CSV文件加载数据: {file_path}")
            return pd.read_csv(file_path, **kwargs)
        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            raise
        except Exception as e:
            logger.error(f"加载CSV文件时出错: {str(e)}")
            raise
    
    def load_from_excel(self, file_path, **kwargs):
        """
        从Excel文件加载数据
        
        参数:
            file_path: Excel文件路径
            **kwargs: pandas.read_excel()的其他参数
            
        返回:
            pandas.DataFrame: 加载的数据
        """
        try:
            logger.info(f"从Excel文件加载数据: {file_path}")
            return pd.read_excel(file_path, **kwargs)
        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            raise
        except Exception as e:
            logger.error(f"加载Excel文件时出错: {str(e)}")
            raise
    
    def fetch_from_api(self, url, params=None, headers=None, api_key=None, save_path=None):
        """
        从API获取数据
        
        参数:
            url: API端点URL
            params: 请求参数
            headers: 请求头
            api_key: API密钥
            save_path: 保存路径，如果提供则保存数据到本地
            
        返回:
            pandas.DataFrame: 获取的数据
        """
        try:
            logger.info(f"从API获取数据: {url}")
            
            # 如果提供了API密钥，则添加到请求头或参数中
            if api_key:
                if headers is None:
                    headers = {}
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            
            # 根据响应内容类型解析数据
            content_type = response.headers.get('Content-Type', '')
            if 'json' in content_type:
                data = pd.json_normalize(response.json())
            elif 'csv' in content_type:
                from io import StringIO
                data = pd.read_csv(StringIO(response.text))
            else:
                # 尝试解析为JSON
                try:
                    data = pd.json_normalize(response.json())
                except:
                    logger.error(f"无法解析API响应，内容类型: {content_type}")
                    raise ValueError(f"无法解析API响应，内容类型: {content_type}")
            
            # 如果提供了保存路径，则保存数据
            if save_path:
                self._save_data(data, save_path)
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"API请求失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"处理API数据时出错: {str(e)}")
            raise
    
    def scrape_website(self, url, selector, headers=None, save_path=None):
        """
        从网站爬取数据
        
        参数:
            url: 网站URL
            selector: CSS选择器，用于定位数据
            headers: 请求头
            save_path: 保存路径，如果提供则保存数据到本地
            
        返回:
            pandas.DataFrame: 爬取的数据
        """
        try:
            logger.info(f"从网站爬取数据: {url}")
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            
            soup = BeautifulSoup(response.text, 'html.parser')
            elements = soup.select(selector)
            
            # 提取数据
            data = []
            for element in elements:
                # 这里的提取逻辑需要根据具体网站结构调整
                item = {
                    'text': element.get_text(strip=True),
                    'url': element.get('href') if element.name == 'a' else None
                }
                data.append(item)
            
            df = pd.DataFrame(data)
            
            # 如果提供了保存路径，则保存数据
            if save_path:
                self._save_data(df, save_path)
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"网页请求失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"爬取网站数据时出错: {str(e)}")
            raise
    
    def _save_data(self, data, file_path):
        """
        保存数据到文件
        
        参数:
            data: 要保存的数据 (pandas.DataFrame)
            file_path: 文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            
            # 根据文件扩展名保存数据
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                data.to_csv(file_path, index=False)
            elif ext in ['.xlsx', '.xls']:
                data.to_excel(file_path, index=False)
            elif ext == '.json':
                data.to_json(file_path, orient='records', force_ascii=False)
            else:
                logger.warning(f"不支持的文件格式: {ext}，默认保存为CSV")
                data.to_csv(file_path, index=False)
            
            logger.info(f"数据已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存数据时出错: {str(e)}")
            raise

# 示例使用
if __name__ == "__main__":
    # 创建数据加载器实例
    loader = DataLoader()
    
