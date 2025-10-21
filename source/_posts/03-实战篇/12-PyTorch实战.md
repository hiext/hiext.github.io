# 第12章：PyTorch实战

## 📝 本章目标
- 掌握PyTorch核心概念
- 学习动态计算图
- 实现自定义模型
- 掌握训练技巧

## 12.1 PyTorch基础

```python
import torch
import torch.nn as nn

# 张量创建
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 3)
z = torch.randn(2, 2)

# 自动求导
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.sum().backward()
print(x.grad)  # dy/dx = 2x
```

## 12.2 神经网络模型

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
```

## 12.3 训练流程

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# 数据加载
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义模型、损失、优化器
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
```

## 12.4 CNN实现

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 9216)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## 12.5 迁移学习

```python
import torchvision.models as models

# 加载预训练模型
resnet = models.resnet18(pretrained=True)

# 冻结参数
for param in resnet.parameters():
    param.requires_grad = False

# 替换最后一层
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# 只训练最后一层
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)
```

## 12.6 模型保存

```python
# 保存整个模型
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# 只保存参数
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))

# 保存训练状态
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')
```

## 📚 本章小结
- ✅ PyTorch张量操作
- ✅ 动态计算图
- ✅ 自定义模型构建
- ✅ 完整训练流程
- ✅ 迁移学习和模型保存

[⬅️ 上一章](./11-TensorFlow实战.md) | [返回目录](../README.md) | [下一章 ➡️](./13-大语言模型应用.md)
