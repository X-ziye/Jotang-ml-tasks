import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 从你刚才的 task2.py 里导入模型类和数据集类
from task2 import SimpleCNN, CatDogDataset, test_raw, test_indices

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载刚才训练好的模型
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
model.eval()

# 2. 准备测试集
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = CatDogDataset(test_raw, test_indices, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ================= 核心任务 1：特征图可视化（看模型在看什么） =================
# 拿一张测试集的图
img, label = test_dataset[0]
img_tensor = img.unsqueeze(0).to(device)  # (1, 3, 32, 32)

# 提取第一层卷积的输出
conv1_out = model.conv1(img_tensor)
conv1_out = model.relu(conv1_out) # 过 ReLU

# 画图
fig, axes = plt.subplots(1, 17, figsize=(20, 2))
# 先把原图画出来（需要反归一化）
img_show = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
axes[0].imshow(img_show)
axes[0].set_title("原图")
axes[0].axis('off')

# 画出前 16 个特征图
for i in range(16):
    feature_map = conv1_out[0, i, :, :].cpu().detach().numpy()
    axes[i+1].imshow(feature_map, cmap='viridis')
    axes[i+1].set_title(f"通道 {i}")
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig('feature_maps.png', dpi=150)
print("特征图已保存为 feature_maps.png")

# ================= 核心任务 2：错误样本分析（抓出分错的图） =================
print("\n开始寻找预测错误的样本...")
wrong_samples = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float().squeeze()
        
        # 找出预测错的
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                # 存下图片、真实标签、预测概率
                wrong_samples.append((images[i].cpu(), labels[i].item(), outputs[i].item()))
                if len(wrong_samples) >= 8: # 找 8 张就够了
                    break
        if len(wrong_samples) >= 8:
            break

# 画错误样本
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for idx, (img_tensor, true_label, pred_prob) in enumerate(wrong_samples):
    ax = axes[idx // 4, idx % 4]
    # 反归一化显示
    img_show = img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
    ax.imshow(img_show)
    
    true_text = "猫" if true_label == 0 else "狗"
    pred_text = "狗" if pred_prob > 0.5 else "猫"
    ax.set_title(f"真实: {true_text}\n误判为: {pred_text}\n概率: {pred_prob:.2f}", color='red')
    ax.axis('off')

plt.tight_layout()
plt.savefig('wrong_samples.png', dpi=150)
print("错误样本已保存为 wrong_samples.png")