# main.py - 项目主入口
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.analysis import DataAnalyzer
from src.visualization import DataVisualizer

def main():
    loader = DataLoader(data_dir="data")
    data = loader.load_from_csv("data.csv")

    preprocessor = DataPreprocessor()
    cleaned_data = preprocessor.clean_data(data)
    processed_data = preprocessor.handle_missing_values(cleaned_data)
    encoded_data = preprocessor.encode_categorical_features(processed_data)
    normalized_data = preprocessor.normalize_numeric_features(encoded_data)

    analyzer = DataAnalyzer()
    correlation_matrix = analyzer.calculate_correlation(normalized_data)
    descriptive_stats = analyzer.descriptive_statistics(normalized_data)  # 需确保 DataAnalyzer 包含此方法

    visualizer = DataVisualizer(output_dir="visualizations")

    # 绘制直方图（指定 bins=30）
    visualizer.create_histogram(
        data=normalized_data,
        column="heart_rate",
        title="Heart Rate Distribution",
        bins=30,  # 显式覆盖默认的 None
        kde=True,
        figsize=(10, 6),
        save_as="heart_rate_histogram"
    )

    # 绘制散点图（使用有效列名）
    visualizer.create_scatter_plot(
        data=normalized_data,
        x="pressure_level",
        y="temperature",
        title="Pressure Level vs Temperature",
        hue="heart_rate",  # 使用存在的数值型列
        palette="viridis",  # 为数值型 hue 设置调色板
        figsize=(10, 6),
        save_as="pressure_vs_temperature_scatter"
    )

    # 绘制热力图（假设 correlation_matrix 正确）
    visualizer.create_heatmap(
        data=correlation_matrix,
        title="Feature Correlation Heatmap",
        annot=True,
        fmt=".2f",  # 只显示 2 位小数
        cmap="viridis",
        figsize=(14, 12),
        xticklabels=5,  # 每 5 个标签显示一次（隐藏重叠标签）
        yticklabels=5,
        save_as="correlation_heatmap_optimized"
    )
if __name__ == "__main__":
    main()