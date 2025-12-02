from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage.morphology import skeletonize


def enhance_binary_crack_image(image_path, output_dir=None):
    """
    针对已二值化图像的裂缝修复与增强流程
    """
    # 步骤1: 读取二值图像
    print("步骤1: 读取二值图像...")
    binary_mask = read_binary_image(image_path)

    # 步骤2: 裂缝修复与增强
    print("步骤2: 裂缝修复与增强...")
    enhanced_mask = crack_repair_enhancement(binary_mask)

    # 步骤3: 质量控制与验证
    print("步骤3: 质量控制...")
    quality_report = quality_control(binary_mask, enhanced_mask)
    print(f"质量报告: {quality_report}")

    # 步骤4: 保存结果
    if output_dir:
        save_enhancement_results(binary_mask, enhanced_mask, image_path, output_dir)

    # # 步骤5: 可视化结果
    # visualize_enhancement_process(binary_mask, enhanced_mask)

    return binary_mask, enhanced_mask


def read_binary_image(image_path):
    """
    读取二值图像并确保格式正确
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    print(f"图像尺寸: {image.shape}")
    print(f"像素值范围: [{image.min()}, {image.max()}]")

    # 确保图像是二值的 (0 和 255 或 0 和 1)
    if image.max() == 1:
        # 已经是0-1格式
        binary_mask = image.astype(bool)
    else:
        # 将图像转换为二值 (0 和 1)
        _, binary_mask = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

    # 确保裂缝是前景(True/1)，背景是背景(False/0)
    # 如果大部分像素是前景，可能需要反转
    foreground_ratio = np.mean(binary_mask)
    print(f"前景像素比例: {foreground_ratio:.4f}")

    if foreground_ratio > 0.5:
        print("检测到可能的前景/背景反转，进行校正...")
        binary_mask = ~binary_mask

    return binary_mask


def crack_repair_enhancement(binary_mask):
    """
    裂缝修复与增强的核心函数
    包含四个主要步骤: 去噪、连接、填充、平滑
    """
    # 复制原始掩膜
    enhanced = binary_mask.copy().astype(np.uint8)

    # 步骤1: 去除孤立噪声点
    print("  - 去除孤立噪声点...")
    enhanced = remove_isolated_noise(enhanced)

    # 步骤2: 连接微小断裂
    print("  - 连接微小断裂...")
    enhanced = connect_small_breaks(enhanced)

    # 步骤3: 填充小孔洞
    print("  - 填充小孔洞...")
    enhanced = fill_small_holes(enhanced)

    # 步骤4: 平滑边界
    print("  - 平滑边界...")
    enhanced = smooth_boundaries(enhanced)

    return enhanced > 0


def remove_isolated_noise(binary_mask, min_size=10):
    """
    步骤1: 去除孤立噪声点
    """
    # 使用连通组件分析找到小区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    # 创建去噪后的掩膜
    denoised = np.zeros_like(binary_mask)

    # 只保留足够大的组件
    for i in range(1, num_labels):  # 跳过背景(0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            denoised[labels == i] = 1

    print(
        f"    原始组件数={num_labels - 1}, 保留组件数={len([s for s in stats[1:, cv2.CC_STAT_AREA] if s >= min_size])}")

    return denoised


def connect_small_breaks(binary_mask, max_gap=5):
    """
    步骤2: 连接微小断裂
    """
    # 方法1: 形态学闭运算连接小断裂
    kernel_size = max(3, max_gap * 2 - 1)  # 确保是奇数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 方法2: 距离变换引导的连接 (更智能)
    connected = distance_based_connection(binary_mask, max_gap)

    # 结合两种方法
    combined = closed | connected

    return combined


def distance_based_connection(binary_mask, max_gap):
    """
    基于距离变换的智能连接方法
    """
    # 计算距离变换
    dist_transform = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # 创建骨架
    skeleton = skeletonize(binary_mask)

    # 在骨架上找到端点
    endpoints = find_endpoints(skeleton)

    # 连接靠近的端点
    connected = connect_nearby_endpoints(binary_mask, endpoints, max_gap)

    return connected


def find_endpoints(skeleton):
    """
    在骨架上找到端点
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # 卷积计算每个点的邻居数量
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

    # 端点只有1个邻居 (10 + 1 = 11)
    endpoints = (neighbors == 11)

    return endpoints


def connect_nearby_endpoints(binary_mask, endpoints, max_gap):
    """
    连接靠近的端点
    """
    connected = binary_mask.copy()
    endpoint_coords = np.argwhere(endpoints)

    # 计算端点之间的距离
    for i, coord1 in enumerate(endpoint_coords):
        for j, coord2 in enumerate(endpoint_coords[i + 1:], i + 1):
            distance = np.sqrt(np.sum((coord1 - coord2) ** 2))

            # 如果端点距离小于阈值，连接它们
            if distance <= max_gap:
                cv2.line(connected, tuple(coord1[::-1]), tuple(coord2[::-1]), 1, 1)

    return connected


def fill_small_holes(binary_mask, max_hole_size=50):
    """
    步骤3: 填充小孔洞
    """
    # 方法1: 形态学闭运算填充小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 方法2: 基于轮廓的孔洞填充
    filled = contour_based_filling(closed, max_hole_size)

    return filled


def contour_based_filling(binary_mask, max_hole_size):
    """
    基于轮廓的孔洞填充方法
    """
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 创建填充后的掩膜
    filled = binary_mask.copy()

    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # 检查是否是孔洞 (有父轮廓)
            if hierarchy[0][i][3] >= 0:
                area = cv2.contourArea(contour)
                # 只填充小孔洞
                if area < max_hole_size:
                    cv2.fillPoly(filled, [contour], 1)

    return filled


def smooth_boundaries(binary_mask, smoothing_iterations=1):
    """
    步骤4: 平滑边界
    """
    smoothed = binary_mask.copy().astype(np.uint8)

    # 使用形态学开运算平滑凸起
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel_open)

    # 使用形态学闭运算平滑凹陷
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel_close)

    # 可选: 使用高斯模糊平滑边界
    if smoothing_iterations > 0:
        blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (5, 5), 0.5)
        smoothed = (blurred > 0.5).astype(np.uint8)

    return smoothed


def quality_control(original_mask, enhanced_mask):
    """
    质量控制与验证
    """
    # 计算各种质量指标
    original_area = np.sum(original_mask)
    enhanced_area = np.sum(enhanced_mask)
    area_change = (enhanced_area - original_area) / original_area if original_area > 0 else 0

    # 连通性改善
    original_components = count_connected_components(original_mask)
    enhanced_components = count_connected_components(enhanced_mask)
    connectivity_improvement = original_components - enhanced_components

    # 边界平滑度
    original_smoothness = calculate_boundary_smoothness(original_mask)
    enhanced_smoothness = calculate_boundary_smoothness(enhanced_mask)
    smoothness_improvement = enhanced_smoothness - original_smoothness

    quality_report = {
        'area_change_percent': round(area_change * 100, 2),
        'connectivity_improvement': connectivity_improvement,
        'smoothness_improvement': round(smoothness_improvement, 4),
        'original_components': original_components,
        'enhanced_components': enhanced_components
    }

    return quality_report


def count_connected_components(binary_mask):
    """计算连通组件数量"""
    num_labels, _ = cv2.connectedComponents(binary_mask.astype(np.uint8))
    return num_labels - 1  # 减去背景


def calculate_boundary_smoothness(binary_mask):
    """计算边界平滑度"""
    # 提取边界
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0

    # 计算所有边界的平滑度 (周长与面积比)
    total_smoothness = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 0:
            # 圆形度: 4π*面积/周长²，值越接近1越平滑
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            total_smoothness += circularity

    return total_smoothness / len(contours)


def save_enhancement_results(binary_mask, enhanced_mask, image_path, output_dir):
    """保存增强结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # # 保存原始二值图像
    # binary_uint8 = binary_mask.astype(np.uint8) * 255
    # cv2.imwrite(os.path.join(output_dir, f"{filename}_original_binary.png"), binary_uint8)

    # 保存增强结果
    enhanced_uint8 = enhanced_mask.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, f"{filename}.png"), enhanced_uint8)

    # # 保存对比图
    # fig = create_enhancement_comparison_figure(binary_mask, enhanced_mask)
    # fig.savefig(os.path.join(output_dir, f"{filename}_enhancement_comparison.png"),
    #             dpi=150, bbox_inches='tight')
    # plt.close(fig)

    print(f"结果已保存到: {output_dir}")


def create_enhancement_comparison_figure(binary_mask, enhanced_mask):
    """创建增强过程对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：整体对比
    axes[0, 0].imshow(binary_mask, cmap='gray')
    axes[0, 0].set_title('原始二值图像')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(enhanced_mask, cmap='gray')
    axes[0, 1].set_title('修复与增强后')
    axes[0, 1].axis('off')

    # 差异可视化
    difference = enhanced_mask.astype(int) - binary_mask.astype(int)
    axes[0, 2].imshow(difference, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 2].set_title('差异图\n(红色=新增, 蓝色=移除)')
    axes[0, 2].axis('off')

    # 第二行：细节对比
    detail_region = find_detail_region(binary_mask | enhanced_mask)
    y1, y2, x1, x2 = detail_region

    # 原始图像细节
    axes[1, 0].imshow(binary_mask[y1:y2, x1:x2], cmap='gray')
    axes[1, 0].set_title('原始二值图 (细节)')
    axes[1, 0].axis('off')

    # 增强后细节
    axes[1, 1].imshow(enhanced_mask[y1:y2, x1:x2], cmap='gray')
    axes[1, 1].set_title('增强后 (细节)')
    axes[1, 1].axis('off')

    # 差异细节
    axes[1, 2].imshow(difference[y1:y2, x1:x2], cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_title('差异图 (细节)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig


def find_detail_region(combined_mask, region_size=150):
    """找到包含裂缝的感兴趣区域"""
    # 找到所有裂缝像素的坐标
    y_coords, x_coords = np.where(combined_mask)

    if len(y_coords) == 0:
        # 如果没有裂缝，返回中心区域
        h, w = combined_mask.shape
        return h // 2 - region_size // 2, h // 2 + region_size // 2, w // 2 - region_size // 2, w // 2 + region_size // 2

    # 计算裂缝的中心点
    center_y = int(np.mean(y_coords))
    center_x = int(np.mean(x_coords))

    # 返回以中心点为中心的区域
    y1 = max(0, center_y - region_size // 2)
    y2 = min(combined_mask.shape[0], center_y + region_size // 2)
    x1 = max(0, center_x - region_size // 2)
    x2 = min(combined_mask.shape[1], center_x + region_size // 2)

    return y1, y2, x1, x2


def visualize_enhancement_process(binary_mask, enhanced_mask):
    """可视化增强过程"""
    plt.figure(figsize=(15, 5))

    # 原始二值图像
    plt.subplot(1, 3, 1)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('原始二值图像')
    plt.axis('off')

    # 修复与增强后
    plt.subplot(1, 3, 2)
    plt.imshow(enhanced_mask, cmap='gray')
    plt.title('修复与增强后')
    plt.axis('off')

    # 差异图
    plt.subplot(1, 3, 3)
    difference = enhanced_mask.astype(int) - binary_mask.astype(int)
    plt.imshow(difference, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('差异图 (红色=新增, 蓝色=移除)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def batch_enhance_binary_cracks(input_dir, output_dir):
    """批量处理二值裂缝图像"""
    # 支持的图像格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in valid_extensions]

    print(f"找到 {len(image_files)} 个二值图像文件")

    all_results = {}

    for i, image_file in enumerate(image_files):
        print(f"\n处理图像 {i + 1}/{len(image_files)}: {image_file}")
        image_path = os.path.join(input_dir, image_file)

        try:
            binary, enhanced = enhance_binary_crack_image(image_path, output_dir)
            all_results[image_file] = {
                'binary': binary,
                'enhanced': enhanced
            }
        except Exception as e:
            print(f"处理失败 {image_file}: {e}")

    return all_results

if __name__ == '__main__':
    # # 处理单张二值图像
    # image_path = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\Standardize_processed_results\\HL1_XY_000.png"
    # binary_mask, enhanced_mask = enhance_binary_crack_image(
    #     image_path,
    #     output_dir="./processed_test"  # 保存结果
    # )

    # 批量处理整个文件夹的二值图像
    input_folder = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\Standardize_processed_results"
    output_folder = "./Enhancement_processed_results"

    results = batch_enhance_binary_cracks(input_folder, output_folder)