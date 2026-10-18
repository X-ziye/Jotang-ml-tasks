import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 解决画图时中文变成方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# ================= 0. 参数设置 =================
hidden_size = 16 
# 学习率改回0.01
learning_rate = 0.01  

# ================= 1. 拿数据 =================
# 生成一千个带噪声的半月形数据
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# 按照 6:2:2 切分训练集、验证集、测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# 把 NumPy 数组转成 PyTorch 的 Tensor 格式
X_train_t = torch.tensor(X_train, dtype=torch.float32)
# 标签原本是 (800,)，这里加个 unsqueeze(1) 变成 (800, 1)，不然算 Loss 的时候维度对不上
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ================= 2. 搭网络 =================
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        # 两层全连接，夹一个 ReLU
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 数据怎么往前流
        z1 = self.fc1(x)          
        a1 = self.relu(z1)         
        z2 = self.fc2(a1)          
        a2 = self.sigmoid(z2)      
        return a2

# 实例化模型
model = MLP(hidden_size=hidden_size)

# 损失函数用二分类交叉熵，优化器用 Adam
criterion = nn.BCELoss()            
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

# ================= 3. 开始训练 =================
epochs = 100
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(epochs):
    # ---- 训练 ----
    model.train()
    optimizer.zero_grad()             # 清空上一轮的梯度，不然会越累加越大
    y_pred = model(X_train_t)         # 前向传播拿到预测值
    loss = criterion(y_pred, y_train_t) # 算一下误差有多大
    loss.backward()                   # 反向传播，让 PyTorch 自动算每个参数的梯度
    optimizer.step()                  # 优化器根据梯度更新参数

    train_acc = ((y_pred > 0.5).float() == y_train_t).float().mean().item()

    # ---- 验证 ----
    model.eval()                      # 切换到测试模式
    with torch.no_grad():             # 验证阶段不需要求导，关掉计算图省内存
        y_val_pred = model(X_val_t)
        val_loss = criterion(y_val_pred, y_val_t)
        val_acc = ((y_val_pred > 0.5).float() == y_val_t).float().mean().item()

    # 把这几轮的指标存下来，待会儿画图用
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print("\n训练完成！")

# ================= 4. 画曲线 =================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Curve (lr={learning_rate})')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Accuracy Curve (lr={learning_rate})')
plt.legend()

plt.tight_layout()
# 文件名带上参数，免得把之前 lr=0.01 的图覆盖掉
plt.savefig(f'loss_accuracy_curves_h{hidden_size}_lr{learning_rate}.png', dpi=150)
print(f"Loss 与 Accuracy 曲线已保存。")

# ================= 5. 画决策边界 =================
def plot_decision_boundary(model, X, y, title, suffix=""):
    # 在整个平面上撒一堆网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 把这些网格点丢给模型，看看模型觉得它们属于哪一类
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid).numpy().reshape(xx.shape)
    
    # 画等高线（背景颜色）和真实的散点
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'blue'])
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='类别 0', edgecolors='k')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='类别 1', edgecolors='k')
    plt.title(title)
    plt.legend()
    plt.savefig(f'decision_boundary_{suffix}.png', dpi=150)
    print(f"决策边界图已保存。")

# 测试集画一下
plot_decision_boundary(model, X_test, y_test, f"MLP 决策边界 (lr={learning_rate})", suffix=f"h{hidden_size}_lr{learning_rate}")

# ================= 6. 混淆矩阵 =================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_t)
    y_test_pred_class = (y_test_pred > 0.5).float()

# 看看真正对了几个，错判了几个
cm = confusion_matrix(y_test_t.numpy(), y_test_pred_class.numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['类别 0', '类别 1'])
disp.plot(cmap='Blues')
plt.title(f"测试集混淆矩阵 (lr={learning_rate})")
plt.savefig(f'confusion_matrix_h{hidden_size}_lr{learning_rate}.png', dpi=150)
print(f"混淆矩阵已保存。")