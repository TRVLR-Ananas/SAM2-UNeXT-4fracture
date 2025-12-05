from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import os


def process_crack_image(image_path, target_width=3, output_dir=None):
    """
    完整的裂缝处理流程：读取 → 二值化 → 标准化宽度

    参数:
        image_path: 输入图像路径
        target_width: 目标裂缝宽度（像素）
        output_dir: 输出目录（可选）
    """
    # 步骤1: 读取图片
    print("步骤1: 读取图片...")
    original_image = read_image(image_path)

    # 步骤2: 二值化处理
    print("步骤2: 二值化处理...")
    binary_mask = binarize_image(original_image)

    # 步骤3: 标准化裂缝宽度
    print("步骤3: 标准化裂缝宽度...")
    standardized_mask = standardize_crack_width(binary_mask, target_width)

    # 保存结果
    if output_dir:
        save_results(original_image, binary_mask, standardized_mask,
                     image_path, output_dir, target_width)

    # 显示结果
    # visualize_results(original_image, binary_mask, standardized_mask, target_width)

    return binary_mask, standardized_mask


def read_image(image_path):
    """步骤1: 读取图片并进行预处理"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    print(f"图像尺寸: {gray.shape}")
    return gray


def binarize_image(image, method='otsu'):
    """步骤2: 二值化处理"""

    if method == 'adaptive':
        # 自适应阈值 - 适合光照不均匀的情况
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'otsu':
        # Otsu自动阈值
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # 手动阈值
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 确保裂缝为白色(255)，背景为黑色(0)
    # 如果大部分像素是白色，则反转
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # 转换为布尔类型 (True/False)
    binary_bool = binary > 0

    print(f"二值化完成 - 裂缝像素比例: {np.mean(binary_bool):.3f}")
    return binary_bool


def standardize_crack_width(binary_mask, target_width=3):
    """步骤3: 标准化裂缝宽度"""

    # 步骤3.1: 骨架化 - 提取裂缝中心线
    print("  - 骨架化处理...")
    skeleton = skeletonize(binary_mask)

    # 步骤3.2: 计算膨胀核大小
    # 目标宽度通常是奇数，这样有明确的中心点
    kernel_size = target_width if target_width % 2 == 1 else target_width + 1

    # 步骤3.3: 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 步骤3.4: 膨胀到目标宽度
    skeleton_uint8 = skeleton.astype(np.uint8) * 255
    standardized = cv2.dilate(skeleton_uint8, kernel, iterations=1)

    # 转换回布尔类型
    standardized_bool = standardized > 0

    # 计算宽度变化
    original_area = np.sum(binary_mask)
    new_area = np.sum(standardized_bool)

    print(f"  - 原始裂缝面积: {original_area}")
    print(f"  - 标准化后面积: {new_area}")
    print(f"  - 面积变化比例: {new_area / original_area:.2f}")

    return standardized_bool


def save_results(original, binary, standardized, image_path, output_dir, target_width):
    """保存处理结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # 保存二值化结果
    # binary_uint8 = binary.astype(np.uint8) * 255
    # cv2.imwrite(os.path.join(output_dir, f"{filename}_binary.png"), binary_uint8)

    # 保存标准化结果
    standardized_uint8 = standardized.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, f"{filename}.png"),
                standardized_uint8)

    # 保存对比图
    # fig = create_comparison_figure(original, binary, standardized, target_width)
    # fig.savefig(os.path.join(output_dir, f"{filename}_comparison.png"),
    #             dpi=150, bbox_inches='tight')
    # plt.close(fig)

    print(f"结果已保存到: {output_dir}")


def create_comparison_figure(original, binary, standardized, target_width):
    """创建结果对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：原始图像和二值化
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('二值化结果')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(standardized, cmap='gray')
    axes[0, 2].set_title(f'标准化宽度 (目标: {target_width}像素)')
    axes[0, 2].axis('off')

    # 第二行：细节对比
    # 找到裂缝区域进行细节展示
    detail_region = find_detail_region(binary)
    y1, y2, x1, x2 = detail_region

    axes[1, 0].imshow(original[y1:y2, x1:x2], cmap='gray')
    axes[1, 0].set_title('原始图像 (细节)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(binary[y1:y2, x1:x2], cmap='gray')
    axes[1, 1].set_title('二值化 (细节)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(standardized[y1:y2, x1:x2], cmap='gray')
    axes[1, 2].set_title('标准化 (细节)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig


def find_detail_region(binary_mask, region_size=100):
    """找到包含裂缝的感兴趣区域"""
    # 找到所有裂缝像素的坐标
    y_coords, x_coords = np.where(binary_mask)

    if len(y_coords) == 0:
        # 如果没有裂缝，返回中心区域
        h, w = binary_mask.shape
        return h // 2 - region_size // 2, h // 2 + region_size // 2, w // 2 - region_size // 2, w // 2 + region_size // 2

    # 计算裂缝的中心点
    center_y = int(np.mean(y_coords))
    center_x = int(np.mean(x_coords))

    # 返回以中心点为中心的区域
    y1 = max(0, center_y - region_size // 2)
    y2 = min(binary_mask.shape[0], center_y + region_size // 2)
    x1 = max(0, center_x - region_size // 2)
    x2 = min(binary_mask.shape[1], center_x + region_size // 2)

    return y1, y2, x1, x2


def visualize_results(original, binary, standardized, target_width):
    """可视化处理结果"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('二值化结果')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(standardized, cmap='gray')
    plt.title(f'标准化宽度: {target_width}像素')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def batch_process_crack_images(input_dir, output_dir, target_width=3):
    """批量处理裂缝图像"""
    # 支持的图像格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in valid_extensions]

    print(f"找到 {len(image_files)} 个图像文件")

    all_results = {}

    for i, image_file in enumerate(image_files):
        print(f"\n处理图像 {i + 1}/{len(image_files)}: {image_file}")
        image_path = os.path.join(input_dir, image_file)

        try:
            binary, standardized = process_crack_image(
                image_path, target_width, output_dir
            )
            all_results[image_file] = {
                'binary': binary,
                'standardized': standardized
            }
        except Exception as e:
            print(f"处理失败 {image_file}: {e}")

    return all_results



if __name__ == '__main__':
    # 处理单张图像
    image_path = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\SourceImageMask\\SourceMask\\HL1_XY_000.png"
    binary_mask, standardized_mask = process_crack_image(
        image_path,
        target_width=5,  # 目标宽度3像素
        output_dir="../../AppData/Roaming/JetBrains/PyCharm2025.1/scratches/processed_test"  # 保存结果
    )


#     # 批量处理整个文件夹
#     input_folder = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\SourceImageMask\\SourceMask"
#     output_folder = "./Standardize_processed_results"
#
#     results = batch_process_crack_images(
#         input_folder,
#         output_folder,
#         target_width=5
#     )