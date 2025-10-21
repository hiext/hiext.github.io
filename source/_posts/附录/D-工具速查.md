# 附录D：工具与框架速查表

## 🐍 Python基础

```python
# 列表推导式
[x**2 for x in range(10) if x % 2 == 0]

# Lambda函数
sorted(data, key=lambda x: x['score'])

# 装饰器
@timer
def my_function():
    pass
```

## 🔢 NumPy速查

```python
import numpy as np

# 创建
np.array([1,2,3])
np.zeros((3,4))
np.arange(0, 10, 2)
np.linspace(0, 1, 5)

# 运算
a + b
np.dot(a, b)
np.matmul(A, B)

# 统计
np.mean(arr)
np.std(arr)
np.argmax(arr)
```

## 📊 Pandas速查

```python
import pandas as pd

# 创建
df = pd.DataFrame(data)
s = pd.Series([1,2,3])

# 读写
pd.read_csv('file.csv')
df.to_csv('file.csv')

# 操作
df.head()
df.describe()
df[df['age'] > 25]
df.groupby('city').mean()
```

## 📈 Matplotlib速查

```python
import matplotlib.pyplot as plt

# 基础绘图
plt.plot(x, y)
plt.scatter(x, y)
plt.bar(categories, values)
plt.hist(data, bins=30)

# 设置
plt.xlabel('X轴')
plt.title('标题')
plt.legend()
plt.grid(True)
```

## 🧠 TensorFlow/Keras速查

```python
from tensorflow import keras

# 模型构建
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 编译
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 预测
model.predict(X_test)
```

## 🔥 PyTorch速查

```python
import torch
import torch.nn as nn

# 张量
x = torch.tensor([1,2,3])
x.to('cuda')

# 模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

# 训练
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 🤗 Transformers速查

```python
from transformers import pipeline, AutoModel

# Pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# 加载模型
model = AutoModel.from_pretrained("bert-base-chinese")
```

## 📚 scikit-learn速查

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 评估
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```

## 🔗 LangChain速查

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# LLM
llm = OpenAI(temperature=0.7)

# Prompt
template = PromptTemplate(
    input_variables=["topic"],
    template="写一篇关于{topic}的文章"
)

# Chain
chain = LLMChain(llm=llm, prompt=template)
result = chain.run("AI")
```

## 🎨 Stable Diffusion速查

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("a cat in a garden").images[0]
```

## 🌐 FastAPI速查

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/items/")
def create_item(item: Item):
    return item

# 运行: uvicorn main:app --reload
```

## 🐳 Docker速查

```bash
# 构建镜像
docker build -t myapp .

# 运行容器
docker run -p 8000:8000 myapp

# 查看容器
docker ps

# 停止容器
docker stop container_id
```

## 📝 Git速查

```bash
# 初始化
git init

# 提交
git add .
git commit -m "message"

# 推送
git push origin main

# 分支
git branch feature
git checkout feature
```

## 🔧 常用命令

### pip包管理
```bash
pip install package
pip install -r requirements.txt
pip freeze > requirements.txt
pip list
pip show package
```

### conda环境
```bash
conda create -n env_name python=3.10
conda activate env_name
conda install package
conda list
conda env export > environment.yml
```

## ⚡ 性能优化技巧

### 1. 向量化操作
```python
# 慢
result = []
for x in data:
    result.append(x ** 2)

# 快
result = np.array(data) ** 2
```

### 2. 使用GPU
```python
# TensorFlow
with tf.device('/GPU:0'):
    model.fit(X, y)

# PyTorch
model = model.to('cuda')
```

### 3. 批处理
```python
# 使用DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)
```

## 🐛 调试技巧

```python
# 断点调试
import pdb; pdb.set_trace()

# 打印形状
print(f"Shape: {tensor.shape}")

# 检查NaN
np.isnan(array).any()

# 内存使用
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
```

[返回目录](../README.md)
