import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载原始数据（不做任何处理）
raw_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=None)

# 2. 定义预处理（只做 Resize, ToTensor, Normalize，方便你看对比）
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. 获取同一张图（假设取第15张，刚好是狗或猫）
img_pil, label = raw_dataset[15] 
img_tensor = transform(img_pil)
# 反归一化，把 [-1, 1] 变回 [0, 1] 才能正常显示
img_processed = img_tensor.numpy().transpose(1, 2, 0) * 0.5 + 0.5

# 4. 画图
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img_pil)
axes[0].set_title(f"预处理前 (原始尺寸)")
axes[0].axis('off')

axes[1].imshow(img_processed)
axes[1].set_title(f"预处理后 (32x32 张量)")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('preprocess_compare.png', dpi=150)
print("图片已成功生成：preprocess_compare.png")