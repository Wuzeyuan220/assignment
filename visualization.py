"""
visualization.py - 数据可视化模块

该模块提供了各种数据可视化功能，包括统计图表、地理信息可视化和交互式可视化等，
帮助直观地展示数据分析结果。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 尝试导入 folium，如果失败则创建一个模拟类
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("folium 模块未安装，地理信息可视化功能将不可用。请安装 folium 以使用这些功能。")
    
    # 创建一个模拟类，用于在 folium 不可用时提供错误信息
    class folium:
        class Map:
            def __init__(self, *args, **kwargs):
                raise ImportError("folium 模块未安装。请安装 folium 以使用地理信息可视化功能。")
        
        class plugins:
            class HeatMap:
                def __init__(self, *args, **kwargs):
                    raise ImportError("folium 模块未安装。请安装 folium 以使用地理信息可视化功能。")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataVisualizer:
    """数据可视化器类，负责创建各种类型的可视化图表"""
    
    def __init__(self, output_dir="visualizations"):
        """
        初始化数据可视化器
        
        参数:
            output_dir: 可视化输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set(style="whitegrid")
    
    def save_figure(self, fig, filename, format='png', dpi=300):
        """
        保存图形到文件
        
        参数:
            fig: matplotlib图形对象
            filename: 文件名（不含扩展名）
            format: 文件格式
            dpi: 分辨率
            
        返回:
            str: 保存的文件路径
        """
        file_path = os.path.join(self.output_dir, f"{filename}.{format}")
        try:
            fig.savefig(file_path, format=format, dpi=dpi, bbox_inches='tight')
            logger.info(f"图形已保存到: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"保存图形时出错: {str(e)}")
            return None
    
    def create_bar_chart(self, data, x, y, title=None, xlabel=None, ylabel=None, 
                         hue=None, palette=None, figsize=(10, 6), save_as=None):
        """
        创建条形图
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            hue: 分组变量
            palette: 调色板
            figsize: 图形大小
            save_as: 保存文件名（不含扩展名）
            
        返回:
            matplotlib.figure.Figure: 生成的图形
        """
        logger.info(f"创建条形图: {title or f'{y} by {x}'}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建条形图
        sns.barplot(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax)
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
        
        # 旋转x轴标签以避免重叠
        if len(data[x].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图形
        if save_as:
            self.save_figure(fig, save_as)
        
        return fig
    
    def create_line_chart(self, data, x, y, title=None, xlabel=None, ylabel=None, 
                          hue=None, palette=None, figsize=(10, 6), save_as=None):
        """
        创建折线图
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            hue: 分组变量
            palette: 调色板
            figsize: 图形大小
            save_as: 保存文件名（不含扩展名）
            
        返回:
            matplotlib.figure.Figure: 生成的图形
        """
        logger.info(f"创建折线图: {title or f'{y} over {x}'}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建折线图
        sns.lineplot(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax)
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
        
        # 旋转x轴标签以避免重叠
        if len(data[x].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图形
        if save_as:
            self.save_figure(fig, save_as)
        
        return fig
    
    def create_scatter_plot(self, data, x, y, title=None, xlabel=None, ylabel=None, 
                            hue=None, size=None, palette=None, figsize=(10, 6), 
                            save_as=None):
        """
        创建散点图
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            hue: 颜色分组变量
            size: 点大小变量
            palette: 调色板
            figsize: 图形大小
            save_as: 保存文件名（不含扩展名）
            
        返回:
            matplotlib.figure.Figure: 生成的图形
        """
        logger.info(f"创建散点图: {title or f'{y} vs {x}'}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建散点图
        sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size, palette=palette, ax=ax)
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
        
        plt.tight_layout()
        
        # 保存图形
        if save_as:
            self.save_figure(fig, save_as)
        
        return fig
    
    def create_histogram(self, data, column, title=None, xlabel=None, ylabel='Frequency', 
                         bins=30,  # 默认分箱数设为 30
                         kde=True, figsize=(10, 6), save_as=None):
        """创建直方图"""
        logger.info(f"创建直方图: {title or f'Distribution of {column}'}")
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(data=data, x=column, bins=bins, kde=kde, ax=ax)  # bins 已明确为整数或默认值
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(column)
        ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        if save_as:
            self.save_figure(fig, save_as)
        
        return fig
    
    def create_box_plot(self, data, x, y, title=None, xlabel=None, ylabel=None, 
                        hue=None, palette=None, figsize=(10, 6), save_as=None):
        """
        创建箱线图
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            hue: 分组变量
            palette: 调色板
            figsize: 图形大小
            save_as: 保存文件名（不含扩展名）
            
        返回:
            matplotlib.figure.Figure: 生成的图形
        """
        logger.info(f"创建箱线图: {title or f'{y} by {x}'}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建箱线图
        sns.boxplot(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax)
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
        
        # 旋转x轴标签以避免重叠
        if len(data[x].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图形
        if save_as:
            self.save_figure(fig, save_as)
        
        return fig
    
#     def create_heatmap(self, data, title=None, annot=True, fmt='.2f', 
#                        cmap='coolwarm', figsize=(10, 8), save_as=None):
#         """
#         创建热力图
        
#         参数:
#             data: 数据DataFrame或相关系数矩阵
#             title: 图表标题
#             annot: 是否显示数值
#             fmt: 数值格式
#             cmap: 颜色映射
#             figsize: 图形大小
#             save_as: 保存文件名（不含扩展名）
            
#         返回:
#             matplotlib.figure.Figure: 生成的图形
#         """
#         logger.info(f"创建热力图: {title or 'Heatmap'}")
        
#         fig, ax = plt.subplots(figsize=figsize)
        
#         # 创建热力图
#         sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
        
#         # 设置标题
#         if title:
#             ax.set_title(title)
        
#         plt.tight_layout()
        
#         # 保存图形
#         if save_as:
#             self.save_figure(fig, save_as)
        
#         return fig
    def create_heatmap(self, data, title=None, xlabel=None, ylabel=None, 
                   annot=True, fmt=".2f", cmap="coolwarm", 
                   figsize=(12, 10), save_as=None, 
                   xticklabels="auto", yticklabels="auto"):
        """创建热力图（优化版）"""
        logger.info(f"创建热力图: {title or 'Feature Correlation Heatmap'}")

        # 关键优化：缩短轴标签（仅保留核心部分）
        if isinstance(data, pd.DataFrame):
            data.columns = data.columns.str.split("_").str[-1]  # 只保留最后一段
            data.index = data.index.str.split("_").str[-1]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, 
                    xticklabels=xticklabels, yticklabels=yticklabels, 
                    ax=ax, cbar_kws={"shrink": 0.8})  # 缩小色条

        # 旋转 x 轴标签避免重叠
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        if title:
            ax.set_title(title, pad=20)  # 增加标题与图的间距
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        plt.tight_layout()
        if save_as:
            self.save_figure(fig, save_as)

        return fig
    
    def create_pie_chart(self, data, values, names, title=None, 
                         colors=None, autopct='%1.1f%%', figsize=(10, 6), 
                         save_as=None):
        """
        创建饼图
        
        参数:
            data: 数据DataFrame
            values: 数值列名
            names: 名称列名
            title: 图表标题
            colors: 颜色列表
            autopct: 百分比格式
            figsize: 图形大小
            save_as: 保存文件名（不含扩展名）
            
        返回:
            matplotlib.figure.Figure: 生成的图形
        """
        logger.info(f"创建饼图: {title or f'Distribution of {values}'}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建饼图
        ax.pie(data[values], labels=data[names], colors=colors, autopct=autopct,
               startangle=90, wedgeprops={'edgecolor': 'w'})
        
        # 设置为圆形
        ax.axis('equal')
        
        # 设置标题
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        # 保存图形
        if save_as:
            self.save_figure(fig, save_as)
        
        return fig
    
    def create_time_series_plot(self, data, date_column, value_column, 
                                title=None, xlabel='Date', ylabel=None, 
                                figsize=(12, 6), rolling_window=None, save_as=None):
        """
        创建时间序列图
        
        参数:
            data: 数据DataFrame
            date_column: 日期列名
            value_column: 数值列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            figsize: 图形大小
            rolling_window: 滚动窗口大小，用于计算移动平均
            save_as: 保存文件名（不含扩展名）
            
        返回:
            matplotlib.figure.Figure: 生成的图形
        """
        logger.info(f"创建时间序列图: {title or f'{value_column} over time'}")
        
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
        # 按日期排序
        data = data.sort_values(date_column)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制时间序列
        ax.plot(data[date_column], data[value_column], label=value_column)
        
        # 如果提供了滚动窗口，计算并绘制移动平均
        if rolling_window:
            data[f'{value_column}_rolling'] = data[value_column].rolling(window=rolling_window).mean()
            ax.plot(data[date_column], data[f'{value_column}_rolling'], 
                    label=f'{value_column} ({rolling_window}-day rolling average)',
                    color='red', linewidth=2)
        
        # 设置标题和标签
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(value_column)
        
        # 添加图例
        ax.legend()
        
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图形
        if save_as:
            self.save_figure(fig, save_as)
        
        return fig
    
    def create_plotly_bar(self, data, x, y, title=None, xlabel=None, ylabel=None, 
                          color=None, barmode='group', orientation='v', 
                          save_as=None):
        """
        创建交互式条形图（使用Plotly）
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            color: 颜色分组变量
            barmode: 条形模式 ('group', 'stack', 'overlay')
            orientation: 方向 ('v' 或 'h')
            save_as: 保存文件名（不含扩展名）
            
        返回:
            plotly.graph_objects.Figure: 生成的图形
        """
        logger.info(f"创建交互式条形图: {title or f'{y} by {x}'}")
        
        # 创建条形图
        fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, 
                     orientation=orientation, title=title)
        
        # 设置轴标签
        if xlabel:
            fig.update_xaxes(title_text=xlabel)
        if ylabel:
            fig.update_yaxes(title_text=ylabel)
        
        # 保存为HTML
        if save_as:
            html_path = os.path.join(self.output_dir, f"{save_as}.html")
            fig.write_html(html_path)
            logger.info(f"交互式条形图已保存到: {html_path}")
        
        return fig
    
    def create_plotly_line(self, data, x, y, title=None, xlabel=None, ylabel=None, 
                           color=None, line_group=None, markers=False, 
                           save_as=None):
        """
        创建交互式折线图（使用Plotly）
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            color: 颜色分组变量
            line_group: 线分组变量
            markers: 是否显示标记
            save_as: 保存文件名（不含扩展名）
            
        返回:
            plotly.graph_objects.Figure: 生成的图形
        """
        logger.info(f"创建交互式折线图: {title or f'{y} over {x}'}")
        
        # 创建折线图
        fig = px.line(data, x=x, y=y, color=color, line_group=line_group, 
                      markers=markers, title=title)
        
        # 设置轴标签
        if xlabel:
            fig.update_xaxes(title_text=xlabel)
        if ylabel:
            fig.update_yaxes(title_text=ylabel)
        
        # 保存为HTML
        if save_as:
            html_path = os.path.join(self.output_dir, f"{save_as}.html")
            fig.write_html(html_path)
            logger.info(f"交互式折线图已保存到: {html_path}")
        
        return fig
    
    def create_plotly_scatter(self, data, x, y, title=None, xlabel=None, ylabel=None, 
                              color=None, size=None, hover_data=None, 
                              trendline=None, save_as=None):
        """
        创建交互式散点图（使用Plotly）
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            color: 颜色分组变量
            size: 点大小变量
            hover_data: 悬停时显示的数据列
            trendline: 趋势线类型 ('ols', 'lowess')
            save_as: 保存文件名（不含扩展名）
            
        返回:
            plotly.graph_objects.Figure: 生成的图形
        """
        logger.info(f"创建交互式散点图: {title or f'{y} vs {x}'}")
        
        # 创建散点图
        fig = px.scatter(data, x=x, y=y, color=color, size=size, 
                         hover_data=hover_data, trendline=trendline, title=title)
        
        # 设置轴标签
        if xlabel:
            fig.update_xaxes(title_text=xlabel)
        if ylabel:
            fig.update_yaxes(title_text=ylabel)
        
        # 保存为HTML
        if save_as:
            html_path = os.path.join(self.output_dir, f"{save_as}.html")
            fig.write_html(html_path)
            logger.info(f"交互式散点图已保存到: {html_path}")
        
        return fig
    
    def create_plotly_box(self, data, x, y, title=None, xlabel=None, ylabel=None, 
                          color=None, notched=False, save_as=None):
        """
        创建交互式箱线图（使用Plotly）
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            color: 颜色分组变量
            notched: 是否使用缺口
            save_as: 保存文件名（不含扩展名）
            
        返回:
            plotly.graph_objects.Figure: 生成的图形
        """
        logger.info(f"创建交互式箱线图: {title or f'{y} by {x}'}")
        
        # 创建箱线图
        fig = px.box(data, x=x, y=y, color=color, notched=notched, title=title)
        
        # 设置轴标签
        if xlabel:
            fig.update_xaxes(title_text=xlabel)
        if ylabel:
            fig.update_yaxes(title_text=ylabel)
        
        # 保存为HTML
        if save_as:
            html_path = os.path.join(self.output_dir, f"{save_as}.html")
            fig.write_html(html_path)
            logger.info(f"交互式箱线图已保存到: {html_path}")
        
        return fig
    
    def create_plotly_heatmap(self, data, x, y, z, title=None, xlabel=None, ylabel=None, 
                              color_continuous_scale='RdBu', save_as=None):
        """
        创建交互式热力图（使用Plotly）
        
        参数:
            data: 数据DataFrame
            x: x轴列名
            y: y轴列名
            z: z值列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            color_continuous_scale: 颜色连续尺度
            save_as: 保存文件名（不含扩展名）
            
        返回:
            plotly.graph_objects.Figure: 生成的图形
        """
        logger.info(f"创建交互式热力图: {title or 'Heatmap'}")
        
        # 创建热力图
        fig = px.density_heatmap(data, x=x, y=y, z=z, 
                                 color_continuous_scale=color_continuous_scale, 
                                 title=title)
        
        # 设置轴标签
        if xlabel:
            fig.update_xaxes(title_text=xlabel)
        if ylabel:
            fig.update_yaxes(title_text=ylabel)
        
        # 保存为HTML
        if save_as:
            html_path = os.path.join(self.output_dir, f"{save_as}.html")
            fig.write_html(html_path)
            logger.info(f"交互式热力图已保存到: {html_path}")
        
        return fig
    
    def create_plotly_time_series(self, data, date_column, value_column, 
                                  title=None, xlabel='Date', ylabel=None, 
                                  color=None, line_group=None, markers=False, 
                                  rolling_window=None, save_as=None):
        """
        创建交互式时间序列图（使用Plotly）
        
        参数:
            data: 数据DataFrame
            date_column: 日期列名
            value_column: 数值列名
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            color: 颜色分组变量
            line_group: 线分组变量
            markers: 是否显示标记
            rolling_window: 滚动窗口大小，用于计算移动平均
            save_as: 保存文件名（不含扩展名）
            
        返回:
            plotly.graph_objects.Figure: 生成的图形
        """
        logger.info(f"创建交互式时间序列图: {title or f'{value_column} over time'}")
        
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
        # 按日期排序
        data = data.sort_values(date_column)
        
        # 创建折线图
        fig = px.line(data, x=date_column, y=value_column, color=color, line_group=line_group, 
                      markers=markers, title=title)
        
        # 如果提供了滚动窗口，计算并绘制移动平均
        if rolling_window:
            data[f'{value_column}_rolling'] = data[value_column].rolling(window=rolling_window).mean()
            fig.add_trace(go.Scatter(x=data[date_column], y=data[f'{value_column}_rolling'], 
                                     mode='lines', name=f'{value_column} ({rolling_window}-day rolling average)',
                                     line=dict(color='red', width=2)))
        
        # 设置轴标签
        if xlabel:
            fig.update_xaxes(title_text=xlabel)
        if ylabel:
            fig.update_yaxes(title_text=ylabel)
        
        # 保存为HTML
        if save_as:
            html_path = os.path.join(self.output_dir, f"{save_as}.html")
            fig.write_html(html_path)
            logger.info(f"交互式时间序列图已保存到: {html_path}")
        
        return fig
    
    def create_geographic_heatmap(self, data, lat_column, lon_column, value_column, 
                                 title=None, center=None, zoom_start=10, 
                                 radius=15, blur=10, save_as=None):
        """
        创建地理热力图
        
        参数:
            data: 数据DataFrame
            lat_column: 纬度列名
            lon_column: 经度列名
            value_column: 数值列名（用于热力图强度）
            title: 图表标题
            center: 地图中心点 [纬度, 经度]
            zoom_start: 初始缩放级别
            radius: 热力点半径
            blur: 热力点模糊程度
            save_as: 保存文件名（不含扩展名）
            
        返回:
            folium.Map: 生成的地图对象
        """
        logger.info(f"创建地理热力图: {title or 'Geographic Heatmap'}")
        
        if not FOLIUM_AVAILABLE:
            logger.error("folium 模块未安装，无法创建地理热力图。")
            raise ImportError("folium 模块未安装。请安装 folium 以使用地理信息可视化功能。")
        
        # 如果未指定中心点，使用数据的平均值
        if center is None:
            center = [data[lat_column].mean(), data[lon_column].mean()]
        
        # 创建地图
        m = folium.Map(location=center, zoom_start=zoom_start)
        
        # 准备热力图数据
        heat_data = [[row[lat_column], row[lon_column], row[value_column]] 
                     for _, row in data.iterrows()]
        
        # 添加热力图层
        HeatMap(heat_data, radius=radius, blur=blur).add_to(m)
        
        # 添加标题
        if title:
            title_html = f'''
                         <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
                         '''
            m.get_root().html.add_child(folium.Element(title_html))
        
        # 保存为HTML
        if save_as:
            html_path = os.path.join(self.output_dir, f"{save_as}.html")
            m.save(html_path)
            logger.info(f"地理热力图已保存到: {html_path}")
        
        return m
    
    def create_geographic_point_map(self, data, lat_column, lon_column, 
                                   popup_column=None, tooltip_column=None, 
                                   color_column=None, color_map=None, 
                                   size_column=None, size_scale=1, 
                                   title=None, center=None, zoom_start=10, 
                                   save_as=None):
        """
        创建地理点地图
        
        参数:
            data: 数据DataFrame
            lat_column: 纬度列名
            lon_column: 经度列名
            popup_column: 弹出窗口内容列名
            tooltip_column: 悬停提示内容列名
            color_column: 颜色列名
            color_map: 颜色映射字典
            size_column: 点大小列名
            size_scale: 大小缩放因子
            title: 图表标题
            center: 地图中心点 [纬度, 经度]
            zoom_start: 初始缩放级别
            save_as: 保存文件名（不含扩展名）
            
        返回:
            folium.Map: 生成的地图对象
        """
        logger.info(f"创建地理点地图: {title or 'Geographic Point Map'}")
        
        if not FOLIUM_AVAILABLE:
            logger.error("folium 模块未安装，无法创建地理点地图。")
            raise ImportError("folium 模块未安装。请安装 folium 以使用地理信息可视化功能。")
        
        # 如果未指定中心点，使用数据的平均值
        if center is None:
            center = [data[lat_column].mean(), data[lon_column].mean()]
        
        # 创建地图
        m = folium.Map(location=center, zoom_start=zoom_start)
        
        # 如果提供了颜色列但没有颜色映射，创建默认映射
        if color_column is not None and color_map is None:
            unique_values = data[color_column].unique()
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                      'darkblue', 'darkgreen', 'cadetblue', 'pink', 'lightblue']
            color_map = {value: colors[i % len(colors)] for i, value in enumerate(unique_values)}
        
        # 添加点
        for _, row in data.iterrows():
            # 设置弹出窗口内容
            popup = None
            if popup_column is not None:
                popup = row[popup_column]
            
            # 设置悬停提示内容
            tooltip = None
            if tooltip_column is not None:
                tooltip = row[tooltip_column]
            
            # 设置点的颜色
            color = 'blue'  # 默认颜色
            if color_column is not None and color_map is not None:
                color = color_map.get(row[color_column], 'blue')
            
            # 设置点的大小
            radius = 5  # 默认大小
            if size_column is not None:
                radius = max(1, min(20, row[size_column] * size_scale))
            
            # 创建标记
            folium.CircleMarker(
                location=[row[lat_column], row[lon_column]],
                radius=radius,
                popup=popup,
                tooltip=tooltip,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)
        
        # 添加标题
        if title:
            title_html = f'''
                         <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
                         '''
            m.get_root().html.add_child(folium.Element(title_html))
        
        # 保存为HTML
        if save_as:
            html_path = os.path.join(self.output_dir, f"{save_as}.html")
            m.save(html_path)
            logger.info(f"地理点地图已保存到: {html_path}")
        
        return m