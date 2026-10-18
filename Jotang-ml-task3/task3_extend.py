import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 数据准备 =================
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
y = y.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= 2. 初始化参数 =================
D_in, H, D_out = 2, 16, 1
np.random.seed(42)
W1 = np.random.randn(D_in, H) * 0.5
b1 = np.zeros((1, H))
W2 = np.random.randn(H, D_out) * 0.5
b2 = np.zeros((1, D_out))

# ================= 3. 纯 NumPy 训练循环 =================
epochs = 200
lr = 0.1
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    # --- 前向传播 ---
    Z1 = np.dot(X_train, W1) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    
    # 计算 Train Loss 和 Acc
    train_loss = -np.mean(y_train * np.log(A2 + 1e-8) + (1 - y_train) * np.log(1 - A2 + 1e-8))
    train_losses.append(train_loss)
    train_accs.append(np.mean((A2 > 0.5) == y_train))
    
    # --- 反向传播 (手算梯度) ---
    N = X_train.shape[0]
    dZ2 = (A2 - y_train) / N
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # --- 最基础的 SGD 更新 ---
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    # --- 验证集评估 ---
    Z1_v = np.dot(X_val, W1) + b1
    A1_v = np.maximum(0, Z1_v)
    Z2_v = np.dot(A1_v, W2) + b2
    A2_v = 1 / (1 + np.exp(-Z2_v))
    val_loss = -np.mean(y_val * np.log(A2_v + 1e-8) + (1 - y_val) * np.log(1 - A2_v + 1e-8))
    val_losses.append(val_loss)
    val_accs.append(np.mean((A2_v > 0.5) == y_val))

# ================= 4. 绘制 Loss 和 Accuracy 曲线 =================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('纯 NumPy 训练 Loss 曲线')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('纯 NumPy 训练 Accuracy 曲线')
plt.legend()

plt.tight_layout()
plt.savefig('numpy_training_curves.png', dpi=150)
print("曲线图已保存为 numpy_training_curves.png")

# ================= 5. 绘制二维决策边界 =================
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid = np.c_[xx.ravel(), yy.ravel()]

# 将整个网格送入训练好的模型
Z1_grid = np.dot(grid, W1) + b1
A1_grid = np.maximum(0, Z1_grid)
Z2_grid = np.dot(A1_grid, W2) + b2
preds = (1 / (1 + np.exp(-Z2_grid))).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'blue'])
plt.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], c='red', label='类别 0', edgecolors='k')
plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], c='blue', label='类别 1', edgecolors='k')
plt.title('纯 NumPy 训练出的决策边界')
plt.legend()
plt.savefig('numpy_decision_boundary.png', dpi=150)
print("决策边界图已保存为 numpy_decision_boundary.png")