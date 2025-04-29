import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_pr_curves(csv_paths, labels, save_path=None):
    """
    读取多个PR曲线CSV数据，绘制在同一图上并显示各自的AP值
    :param csv_paths: CSV文件路径列表
    :param labels: 每个曲线对应的标签列表
    :param save_path: 图片保存路径（None则不保存）
    """
    # 参数校验
    if len(csv_paths) != len(labels):
        raise ValueError("csv_paths and labels must have the same length")
    
    # 创建画布
    plt.figure(figsize=(10, 6), dpi=100)
    
    # 颜色循环配置
    colors = plt.cm.get_cmap('tab10', len(csv_paths))
    
    # 遍历每个CSV文件
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        # 读取数据
        df = pd.read_csv(csv_path)
        
        # 提取数据列
        recall = df.iloc[:, 0].values  # 第一列是Recall
        precision_cols = df.columns[1:]  # 后续列是各类Precision
        
        # 计算平均Precision和AP
        avg_precision = df[precision_cols].mean(axis=1).values
        ap = np.trapz(avg_precision, recall)  # 使用梯形法则计算曲线下面积
        
        # 绘制曲线
        plt.plot(recall, avg_precision,
                 color=colors(i),
                 linewidth=2,
                 label=f'{label} ')
    
    # 图表装饰
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Comparison of Precision-Recall Curves', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(loc='best')  # 自动选择最佳位置
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Saved comparison plot to: {save_path}')
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 需要对比的文件路径和对应标签
    csv_paths = [
        R"F:\zyk\yolov9-main\runs\val\exp65\PR_curve_data.csv",
        R"F:\zyk\yolov9-main\runs\val\exp66\PR_curve_data.csv",
    
    ]
    labels = [
        "SlideLoss",
        "v9c",
    ]
    
    # 生成对比图
    plot_multiple_pr_curves(
        csv_paths=csv_paths,
        labels=labels,
        save_path=R"F:\zyk\PR_figure\pr_curve_comparison.pdf"
    )