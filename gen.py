import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNeXT import SAM2UNeXT
import torchvision.transforms as transforms
from PIL import Image
from StandardizeFracture import batch_process_crack_images

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam2-unext")
parser.add_argument("--generate_image_path", type=str, required=True,
                    help="path to the image files for generating")
parser.add_argument("--save_path", type=str, required=True,
                    help="path to save the generated masks")
parser.add_argument("--stand_save_path", type=str, required=True,
                    help="path to save the standardize generated masks")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 直定义一个简单的数据加载类
class GenDataset:
    def __init__(self, image_root, size=1024):
        # 获取所有图像文件
        self.images = []
        for f in os.listdir(image_root):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                self.images.append(os.path.join(image_root, f))
        self.images = sorted(self.images)

        # 使用dataset.py中相同的转换
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.size = len(self.images)
        self.index = 0

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def load_data(self):
        if self.index >= self.size:
            return None, None, None

        # 加载图像
        image = self.rgb_loader(self.images[self.index])
        original_size = image.size  # (width, height)

        # 应用预处理变换
        image_tensor = self.transform(image).unsqueeze(0)

        # 创建一个假的ground truth（全零数组）
        # 为了兼容原代码返回一个与原始图像大小相同的零数组
        gt = np.zeros((original_size[1], original_size[0]), dtype=np.float32)

        # 获取文件名
        name = os.path.basename(self.images[self.index])

        self.index += 1

        return image_tensor, gt, name

test_loader = GenDataset(args.generate_image_path, 1024)
model = SAM2UNeXT().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)

for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        image = image.to(device)
        res = model(image)
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        # If you want to binarize the prediction results, please uncomment the following three lines.
        # Note that this action will affect the calculation of evaluation metrics.
        threshold_value = 0.788
        res[res >= int(255 * threshold_value)] = 255
        res[res < int(255 * threshold_value)] = 0
        file_name = os.path.basename(name)
        print("Saving " + file_name)
        imageio.imsave(os.path.join(args.save_path, file_name[:-4] + ".png"), res)

target_width=1  # 标准化裂缝宽度
batch_process_crack_images(
    args.save_path,
    args.stand_save_path,
    target_width
)


print(f"All {test_loader.size} images processed!")