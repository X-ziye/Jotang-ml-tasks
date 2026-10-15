import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ================= 0. 全局配置 =================
# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 这里做对照实验：16是基线，2是欠拟合，128是过拟合
hidden_size = 128 

# ================= 1. 数据生成与划分 =================
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# 划分 6:2:2
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# 转换为 PyTorch 张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ================= 2. 定义 MLP 模型 =================
class MLP(nn.Module):
    # 增加 hidden_size 参数
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        # 使用传入的 hidden_size
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z1 = self.fc1(x)          
        a1 = self.relu(z1)         
        z2 = self.fc2(a1)          
        a2 = self.sigmoid(z2)      
        return a2

# 实例化模型，传入 hidden_size
model = MLP(hidden_size=hidden_size)

criterion = nn.BCELoss()            
optimizer = optim.Adam(model.parameters(), lr=0.01)  

# ================= 3. 训练与验证 =================
epochs = 100
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()             
    y_pred = model(X_train_t)         
    loss = criterion(y_pred, y_train_t) 
    loss.backward()                   
    optimizer.step()                  

    train_acc = ((y_pred > 0.5).float() == y_train_t).float().mean().item()

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t)
        val_loss = criterion(y_val_pred, y_val_t)
        val_acc = ((y_val_pred > 0.5).float() == y_val_t).float().mean().item()

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print("\n训练完成！")

# ================= 4. 绘制 Loss 与 Accuracy 曲线 =================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
#  文件名带上 hidden_size
plt.savefig(f'loss_accuracy_curves_h{hidden_size}.png', dpi=150)
print(f"Loss 与 Accuracy 曲线已保存为 loss_accuracy_curves_h{hidden_size}.png")

# ================= 5. 绘制二维决策边界 =================
#  函数增加 suffix 参数
def plot_decision_boundary(model, X, y, title, suffix=""):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid).numpy().reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'blue'])
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='类别 0', edgecolors='k')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='类别 1', edgecolors='k')
    plt.title(title)
    plt.legend()
    #  文件名带上 suffix
    plt.savefig(f'decision_boundary_{suffix}.png', dpi=150)
    print(f"决策边界图已保存为 decision_boundary_{suffix}.png")

#  调用时传入后缀
plot_decision_boundary(model, X_test, y_test, f"MLP 决策边界 (测试集, 隐藏层={hidden_size})", suffix=f"h{hidden_size}")

# ================= 6. 模型保存与加载 =================
#  文件名带上 hidden_size
torch.save(model.state_dict(), f'mlp_model_h{hidden_size}.pth')
print(f"模型已保存为 mlp_model_h{hidden_size}.pth")

loaded_model = MLP(hidden_size=hidden_size)
loaded_model.load_state_dict(torch.load(f'mlp_model_h{hidden_size}.pth'))
loaded_model.eval()
print("模型加载成功！")

# ================= 7. 混淆矩阵 =================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_t)
    y_test_pred_class = (y_test_pred > 0.5).float()

cm = confusion_matrix(y_test_t.numpy(), y_test_pred_class.numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['类别 0', '类别 1'])
disp.plot(cmap='Blues')
plt.title(f"测试集混淆矩阵 (隐藏层={hidden_size})")
#  文件名带上 hidden_size
plt.savefig(f'confusion_matrix_h{hidden_size}.png', dpi=150)
print(f"混淆矩阵已保存为 confusion_matrix_h{hidden_size}.png")