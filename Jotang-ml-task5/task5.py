import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载模型和预处理
device = "cuda" if torch.cuda.is_available() else "cpu"
# 使用较小的 RN50 模型 (ResNet-50)
model, preprocess = clip.load("RN50", device=device)
print(f"CLIP 模型加载成功！使用设备: {device}")

# 2. 准备 4 张图片（自动从网络获取）
urls = [
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400", # 猫
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",    # 狗
    "https://images.unsplash.com/photo-1559825481-12a05cc00344?w=400",    # 飞机
    "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400"  # 汽车
]
images = []
for url in urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(img)

image_input = torch.stack([preprocess(img) for img in images]).to(device)

# 3. 准备 4 段文本描述 (故意加入一句错误的描述)
texts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of an airplane",
    "a red car" # 故意描述错颜色，原图是黑/灰色的汽车
]
text_input = clip.tokenize(texts).to(device)

# 4. 提取特征
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)

print(f"图像特征原始形状: {image_features.shape}")
print(f"文本特征原始形状: {text_features.shape}")

# 5. L2 归一化
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 6. 计算余弦相似度矩阵 (点积)
similarity = (image_features @ text_features.T).cpu().numpy()

# 7. 绘制 4x4 热力图
plt.figure(figsize=(8, 6))
plt.imshow(similarity, cmap='Blues')
plt.colorbar(label='Cosine Similarity')
plt.title("CLIP 图文相似度矩阵")
plt.xlabel("Text Descriptions")
plt.ylabel("Images")
plt.xticks(ticks=np.arange(4), labels=texts, rotation=45, ha='right')
plt.yticks(ticks=np.arange(4), labels=['Image 1 (Cat)', 'Image 2 (Dog)', 'Image 3 (Plane)', 'Image 4 (Car)'])

# 把数值标在格子里
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig('clip_similarity_matrix.png', dpi=150)
print("相似度热力图已保存为 clip_similarity_matrix.png")