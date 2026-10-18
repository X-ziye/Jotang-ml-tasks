import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# 数据准备
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_full = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_full = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# 干净的自定义 Dataset，彻底解决越界问题
class CatDogDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        # CIFAR-10 的 3 是猫，5 是狗。映射为 0 和 1
        new_label = 0 if label == 3 else 1
        return img, new_label
        
    def __len__(self):
        return len(self.indices)

# 找出猫狗
train_indices = [i for i, (_, label) in enumerate(train_full) if label in [3, 5]]
test_indices = [i for i, (_, label) in enumerate(test_full) if label in [3, 5]]

train_dataset = CatDogDataset(train_full, train_indices)
test_dataset = CatDogDataset(test_full, test_indices)

# 划分验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ================= 定义全连接网络 (MLP) =================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1) # 展平 32x32x3 -> 3072
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练 5 个 Epoch
for epoch in range(5):
    model.train()
    for inputs, labels in train_loader:
        labels = labels.unsqueeze(1).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.unsqueeze(1).float()
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"MLP Epoch {epoch+1} | Val Acc: {correct/total:.4f}")