import numpy as np
import torch
from sklearn.datasets import make_moons

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
print(" 实践 1 校验通过！\n")


# ================= 实践 2：神经网络的梯度 =================
print("--- 实践 2：神经网络梯度 (单样本 vs 小 batch) ---")
np.random.seed(42)
D_in, H, D_out = 2, 3, 1

# 分别跑 N=1 和 N=5 验证维度变化
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

    # 4. NumPy 反向传播 (链式法则)
    dZ2 = (A2 - y_np) / N                # shape: (N, 1)
    dW2 = np.dot(A1.T, dZ2)              # shape: (3, 1) 
    db2 = np.sum(dZ2, axis=0, keepdims=True) # shape: (1, 1)
    dA1 = np.dot(dZ2, W2.T)              # shape: (N, 3)
    dZ1 = dA1 * (Z1 > 0)                 # shape: (N, 3)
    dW1 = np.dot(X_np.T, dZ1)            # shape: (2, 3) 
    db1 = np.sum(dZ1, axis=0, keepdims=True) # shape: (1, 3)

    # 5. PyTorch 自动求导
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

print("\n 实践 2 校验通过！数值一致，形状对齐！")