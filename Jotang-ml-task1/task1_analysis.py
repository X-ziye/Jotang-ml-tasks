import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据并划分
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# 转换成 Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ---------- 画原始数据图 ----------
plt.figure(figsize=(6, 5))
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='类别 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='类别 1', alpha=0.6)
plt.title("原始 make_moons 数据分布")
plt.legend()
plt.savefig('raw_data.png', dpi=150)

# ---------- 现场快速训练一个 H=16 的模型 ----------
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

model = MLP(16)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练 100 轮
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

# ---------- 挑出错误样本并画图 ----------
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t)
    pred_labels = (y_pred > 0.5).float()
    
# 找出预测错的索引
wrong_indices = np.where(pred_labels.numpy() != y_test_t.numpy())[0]
print(f"测试集共 {len(y_test)} 个样本，错了 {len(wrong_indices)} 个。")

plt.figure(figsize=(6, 5))
# 用淡色画所有背景点
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.2)
# 用绿圈圈出预测错的地方
plt.scatter(X_test[wrong_indices, 0], X_test[wrong_indices, 1], 
            facecolors='none', edgecolors='green', s=150, linewidths=2, label='预测错误样本')
plt.title("模型预测错误的样本 (绿圈)")
plt.legend()
plt.savefig('wrong_samples.png', dpi=150)
print("错误样本图已保存为 wrong_samples.png")