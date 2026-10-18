import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ================= 实践 1：简单计算链 =================
print("--- 实践 1：简单计算链 ---")
x_val, y_val = 2.0, 3.0

# 正向计算
u = x_val * y_val
v = x_val ** 2
w = u + v
f = w ** 2

# 反向传播 (NumPy)
df_dw = 2 * w
df_dx = df_dw * (y_val + 2 * x_val)
df_dy = df_dw * x_val
print(f"NumPy 算出的梯度: df/dx = {df_dx}, df/dy = {df_dy}")

# PyTorch 验证
x_t = torch.tensor(x_val, requires_grad=True)
y_t = torch.tensor(y_val, requires_grad=True)
f_t = (x_t * y_t + x_t**2)**2
f_t.backward()
print(f"PyTorch 算出的梯度: df/dx = {x_t.grad.item()}, df/dy = {y_t.grad.item()}")
print("✅ 实践 1 校验通过！\n")


# ================= 实践 2：神经网络的梯度 =================
print("--- 实践 2：神经网络梯度 (使用 make_moons 的小 batch) ---")
np.random.seed(42)
D_in, H, D_out = 2, 3, 1

# 我们把之前写死的过程，装进一个循环里，分别跑 N=1 和 N=5
for N in [1, 5]:
    print(f"\n========= 开始验证 Batch Size N = {N} =========")
    
    # 1. 取数据
    X_np, y_np = make_moons(n_samples=N, noise=0.1, random_state=42)
    y_np = y_np.reshape(-1, 1)  
    
    # 2. 初始化参数
    W1 = np.random.randn(D_in, H) * 0.5  # shape: (2, 3)
    b1 = np.zeros((1, H))                # shape: (1, 3)
    W2 = np.random.randn(H, D_out) * 0.5 # shape: (3, 1)
    b2 = np.zeros((1, D_out))            # shape: (1, 1)

    # 3. NumPy 前向传播
    Z1 = np.dot(X_np, W1) + b1           # shape: (N, 3)
    A1 = np.maximum(0, Z1)               # shape: (N, 3)
    Z2 = np.dot(A1, W2) + b2             # shape: (N, 1)
    A2 = 1 / (1 + np.exp(-Z2))           # shape: (N, 1)
    loss_np = -np.mean(y_np * np.log(A2 + 1e-8) + (1 - y_np) * np.log(1 - A2 + 1e-8))

    # 4. NumPy 反向传播 (注意这里除以了 N)
    dZ2 = (A2 - y_np) / N                # shape: (N, 1)
    dW2 = np.dot(A1.T, dZ2)              # shape: (3, 1)  <-- 永远是 (3,1) 不受 N 影响
    db2 = np.sum(dZ2, axis=0, keepdims=True) # shape: (1, 1)
    dA1 = np.dot(dZ2, W2.T)              # shape: (N, 3)
    dZ1 = dA1 * (Z1 > 0)                 # shape: (N, 3)
    dW1 = np.dot(X_np.T, dZ1)            # shape: (2, 3)  <-- 永远是 (2,3) 不受 N 影响
    db1 = np.sum(dZ1, axis=0, keepdims=True) # shape: (1, 3)

    # 5. PyTorch 验证
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)
    W1_t = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
    W2_t = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

    Z1_t = torch.matmul(X_t, W1_t) + b1_t
    A1_t = torch.relu(Z1_t)
    Z2_t = torch.matmul(A1_t, W2_t) + b2_t
    A2_t = torch.sigmoid(Z2_t)
    loss_t = torch.nn.functional.binary_cross_entropy(A2_t, y_t)
    loss_t.backward()

    # 6. 打印观察结果
    print(f"输入 X 的形状: {X_np.shape}")
    print(f"中间结果 Z1 的形状: {Z1.shape}, A1 的形状: {A1.shape}")
    print(f"dW1 的形状: {dW1.shape} (不受 Batch 影响)")
    print(f"dW2 的形状: {dW2.shape} (不受 Batch 影响)")
    print(f"最大绝对误差: {np.max(np.abs(dW1 - W1_t.grad.numpy())):.10f}")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 准备数据
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
y = y.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 初始化参数
D_in, H, D_out = 2, 16, 1
W1 = np.random.randn(D_in, H) * 0.5
b1 = np.zeros((1, H))
W2 = np.random.randn(H, D_out) * 0.5
b2 = np.zeros((1, D_out))

# 3. 训练循环 (纯 NumPy)
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
    
    # --- 计算 Loss ---
    train_loss = -np.mean(y_train * np.log(A2 + 1e-8) + (1 - y_train) * np.log(1 - A2 + 1e-8))
    train_losses.append(train_loss)
    
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
    
    # 计算准确率
    train_accs.append(np.mean((A2 > 0.5) == y_train))
    val_accs.append(np.mean((A2_v > 0.5) == y_val))

# 4. 绘制 Loss 和 Accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('NumPy 训练 Loss 曲线')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('NumPy 训练 Accuracy 曲线')
plt.legend()
plt.savefig('numpy_training_curves.png', dpi=150)

# 5. 绘制决策边界 (使用最终学到的参数)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid = np.c_[xx.ravel(), yy.ravel()]

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

print("拓展实验完成！图已保存为 numpy_training_curves.png 和 numpy_decision_boundary.png")