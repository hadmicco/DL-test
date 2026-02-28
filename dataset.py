import os
import torch
import rasterio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class GlobalGeoTreeDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, is_train=True):
        """
        GlobalGeoTree 遥感影像数据集加载器
        :param data_dir: 存放 .tif 影像的根目录
        :param csv_file: 包含影像文件名和对应树种标签的 CSV 文件
        :param transform: 数据增强操作 (如 Albumentations)
        :param is_train: 区分训练集还是验证集/测试集
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # 你的盲点往往在这里：没有核对标签文件格式。假设这里有一个 CSV 映射。
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"致命错误：找不到标签文件 {csv_file}。请先准备好数据！")
            
        self.metadata = pd.read_csv(csv_file)
        self.image_names = self.metadata['image_filename'].values
        self.labels = self.metadata['tree_species_id'].values

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 1. 拼接影像真实路径
        img_name = os.path.join(self.data_dir, self.image_names[idx])
        
        # 2. 读取遥感影像 (强烈建议使用 rasterio 而不是 cv2/PIL，因为它能处理多光谱和极化雷达)
        try:
            with rasterio.open(img_name) as src:
                # 遥感数据通常自带多个波段 (B, G, R, NIR 等)
                # rasterio 读取出来的 shape 默认为 (Channels, Height, Width)
                image = src.read()
                
                # 转换为 (Height, Width, Channels) 以便后续的数据增强库 (如 Albumentations) 处理
                image = np.transpose(image, (1, 2, 0))
        except Exception as e:
            raise RuntimeError(f"读取影像 {img_name} 失败，检查你的数据是否损坏！错误信息: {e}")

        # 3. 遥感影像硬核预处理 (这是那两篇论文能跑出结果的隐形前提)
        # Sentinel-2 数据通常需要除以 10000 进行反射率归一化
        image = image.astype(np.float32) / 10000.0
        
        # 4. 数据增强 (如果你打算引入论文二中的抗干扰能力，这一步必不可少)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # 如果没有使用 Albumentations 自动转换张量，手动转回 PyTorch 需要的 (C, H, W)
        if not isinstance(image, torch.Tensor):
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()

        # 5. 获取标签
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# --- 代码可用性测试（立刻执行这段代码来检验你的环境） ---
if __name__ == '__main__':
    print("环境探针启动... 如果这段代码报错，说明你的工程基础设施还没建好。")
    # 假设你随便捏造了一张 (10通道, 128, 128) 的张量，模拟加载出来的数据
    dummy_image = torch.randn(10, 128, 128) 
    dummy_label = torch.tensor(3)
    
    print(f"成功模拟加载数据！\n影像维度: {dummy_image.shape} \n标签类别: {dummy_label.item()}")
    print("下一步：准备用这段代码去吞噬真实的 GlobalGeoTree 数据。")