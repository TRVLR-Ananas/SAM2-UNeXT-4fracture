from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def calculate_black_white_ratio(image_path, white_threshold=200, black_threshold=50):
    """
    计算单张图片的黑色和白色像素占比

    参数:
    - image_path: 图片路径
    - white_threshold: 白色阈值，大于此值的像素被认为是白色
    - black_threshold: 黑色阈值，小于此值的像素被认为是黑色

    返回:
    - white_ratio: 白色像素占比
    - black_ratio: 黑色像素占比
    """
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"无法读取图片: {image_path}")
        return 0, 0

    # 获取图片尺寸
    height, width = img.shape
    total_pixels = height * width

    # 统计白色和黑色像素
    white_pixels = np.sum(img >= white_threshold)
    black_pixels = np.sum(img <= black_threshold)

    # 计算占比
    white_ratio = white_pixels / total_pixels
    black_ratio = black_pixels / total_pixels

    return white_ratio, black_ratio


def analyze_folder_images(folder_path, white_threshold=200, black_threshold=50):
    """
    分析文件夹中的所有图片

    参数:
    - folder_path: 文件夹路径
    - white_threshold: 白色阈值
    - black_threshold: 黑色阈值
    """
    # 支持的图片格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    # 收集所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
        image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"在文件夹 {folder_path} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")
    print("-" * 80)
    print(f"{'文件名':<30} {'白色占比(%)':<15} {'黑色占比(%)':<15} {'备注':<20}")
    print("-" * 80)

    # 统计数据
    white_ratios = []
    black_ratios = []
    file_names = []

    for img_path in image_files:
        try:
            white_ratio, black_ratio = calculate_black_white_ratio(
                str(img_path), white_threshold, black_threshold
            )

            # 存储数据
            white_ratios.append(white_ratio * 100)  # 转换为百分比
            black_ratios.append(black_ratio * 100)
            file_names.append(img_path.name)

            # 判断图像类型
            if white_ratio > 0.7:
                image_type = "白色为主"
            elif black_ratio > 0.7:
                image_type = "黑色为主"
            elif abs(white_ratio - black_ratio) < 0.2:
                image_type = "黑白均衡"
            else:
                image_type = "混合"

            print(
                f"{img_path.name:<30} {white_ratio * 100:>8.2f}%      {black_ratio * 100:>8.2f}%      {image_type:<20}")

        except Exception as e:
            print(f"处理图片 {img_path.name} 时出错: {str(e)}")

    # 打印汇总统计
    print("-" * 80)
    if white_ratios:
        print(f"\n汇总统计:")
        print(f"平均白色占比: {np.mean(white_ratios):.2f}%")
        print(f"平均黑色占比: {np.mean(black_ratios):.2f}%")
        print(f"白色占比最高: {np.max(white_ratios):.2f}% ({file_names[np.argmax(white_ratios)]})")
        print(f"黑色占比最高: {np.max(black_ratios):.2f}% ({file_names[np.argmax(black_ratios)]})")
        print(f"白色占比最低: {np.min(white_ratios):.2f}% ({file_names[np.argmin(white_ratios)]})")
        print(f"黑色占比最低: {np.min(black_ratios):.2f}% ({file_names[np.argmin(black_ratios)]})")

    return file_names, white_ratios, black_ratios


def visualize_results(file_names, white_ratios, black_ratios, show_top_n=10):
    """
    可视化结果

    参数:
    - file_names: 文件名列表
    - white_ratios: 白色占比列表
    - black_ratios: 黑色占比列表
    - show_top_n: 显示前N个结果
    """
    # 限制显示数量
    n = min(show_top_n, len(file_names))

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 第一个子图：前N个图片的占比
    indices = range(n)
    bar_width = 0.35

    axes[0].bar(indices, white_ratios[:n], bar_width, label='白色占比', color='lightgray')
    axes[0].bar([i + bar_width for i in indices], black_ratios[:n], bar_width, label='黑色占比', color='black')

    axes[0].set_xlabel('图片索引')
    axes[0].set_ylabel('占比 (%)')
    axes[0].set_title(f'前 {n} 张图片的黑色白色占比')
    axes[0].set_xticks([i + bar_width / 2 for i in indices])
    axes[0].set_xticklabels([f'Img{i + 1}' for i in indices], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 第二个子图：所有图片的占比分布
    axes[1].hist(white_ratios, bins=20, alpha=0.5, label='白色占比分布', color='lightgray')
    axes[1].hist(black_ratios, bins=20, alpha=0.5, label='黑色占比分布', color='black')
    axes[1].set_xlabel('占比 (%)')
    axes[1].set_ylabel('图片数量')
    axes[1].set_title('黑色白色占比分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印所有图片的平均值
    print(f"\n所有 {len(file_names)} 张图片的平均值:")
    print(f"平均白色占比: {np.mean(white_ratios):.2f}%")
    print(f"平均黑色占比: {np.mean(black_ratios):.2f}%")

    # 计算其他颜色（非黑非白）的占比
    other_ratios = [100 - white - black for white, black in zip(white_ratios, black_ratios)]
    print(f"平均其他颜色占比: {np.mean(other_ratios):.2f}%")

    # 判断整体趋势
    avg_white = np.mean(white_ratios)
    avg_black = np.mean(black_ratios)

    if avg_white > avg_black * 1.5:
        print("\n整体趋势: 白色占主导")
    elif avg_black > avg_white * 1.5:
        print("\n整体趋势: 黑色占主导")
    else:
        print("\n整体趋势: 黑白相对均衡")


def main():
    """
    主函数：用户交互式界面
    """
    print("图片黑白占比分析工具")
    print("=" * 50)

    # 获取用户输入
    folder_path = input("请输入图片文件夹路径: ").strip()

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在!")
        return

    # 获取阈值设置（可选）
    use_custom_threshold = input("是否使用自定义阈值？(y/n, 默认n): ").strip().lower()

    white_threshold = 200
    black_threshold = 50

    if use_custom_threshold == 'y':
        try:
            white_threshold = int(input("请输入白色阈值 (0-255, 默认200): ") or 200)
            black_threshold = int(input("请输入黑色阈值 (0-255, 默认50): ") or 50)
        except ValueError:
            print("输入错误，使用默认阈值")

    # 分析图片
    try:
        file_names, white_ratios, black_ratios = analyze_folder_images(
            folder_path, white_threshold, black_threshold
        )

        if file_names:
            # 询问是否可视化结果
            show_plot = input("\n是否显示可视化图表？(y/n, 默认y): ").strip().lower()
            if show_plot != 'n':
                visualize_results(file_names, white_ratios, black_ratios)

            # 询问是否保存结果到文件
            save_csv = input("\n是否保存结果到CSV文件？(y/n, 默认n): ").strip().lower()
            if save_csv == 'y':
                csv_path = input("请输入CSV文件路径 (默认: 结果.csv): ").strip() or "结果.csv"
                save_to_csv(csv_path, file_names, white_ratios, black_ratios)
                print(f"结果已保存到: {csv_path}")

    except Exception as e:
        print(f"分析过程中出错: {str(e)}")


def save_to_csv(csv_path, file_names, white_ratios, black_ratios):
    """
    保存结果到CSV文件
    """
    import csv

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['文件名', '白色占比(%)', '黑色占比(%)', '其他颜色占比(%)'])

        for name, white, black in zip(file_names, white_ratios, black_ratios):
            other = 100 - white - black
            writer.writerow([name, f"{white:.2f}", f"{black:.2f}", f"{other:.2f}"])


# 简单使用方式（直接调用函数）
def quick_analyze(folder_path, white_threshold=200, black_threshold=50, show_plot=True):
    """
    快速分析函数

    使用示例:
    quick_analyze("path/to/your/images", white_threshold=200, black_threshold=50)
    """
    print(f"正在分析文件夹: {folder_path}")
    file_names, white_ratios, black_ratios = analyze_folder_images(
        folder_path, white_threshold, black_threshold
    )

    if show_plot and file_names:
        visualize_results(file_names, white_ratios, black_ratios)

    return file_names, white_ratios, black_ratios


if __name__ == "__main__":
    # 方式1: 使用交互式界面
    main()

    # 方式2: 直接调用（取消注释下面一行，替换为你的文件夹路径）
    # quick_analyze("path/to/your/images", show_plot=True)