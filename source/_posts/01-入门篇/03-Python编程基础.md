# 第3章：Python编程基础

## 📝 本章目标
- 掌握Python核心语法
- 学习NumPy、Pandas数据处理
- 掌握Matplotlib可视化
- 完成数据分析实战

## 3.1 Python核心语法

```python
# 变量与数据类型
name = "张三"
age = 25
scores = [85, 90, 78]
student = {"name": "李四", "age": 20}

# 控制流程
if age >= 18:
    print("成年人")
else:
    print("未成年")

for score in scores:
    print(f"分数: {score}")

# 函数定义
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# 类定义
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"我是{self.name}，{self.age}岁"
```

## 3.2 NumPy数组计算

```python
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

# 数组运算
print(arr * 2)
print(matrix.T)  # 转置
print(np.mean(arr))  # 平均值

# 索引和切片
print(arr[1:3])
print(matrix[0, :])
```

## 3.3 Pandas数据处理

```python
import pandas as pd

# 创建DataFrame
df = pd.DataFrame({
    'name': ['张三', '李四', '王五'],
    'age': [25, 30, 35],
    'salary': [8000, 12000, 15000]
})

# 数据操作
print(df.head())
print(df[df['age'] > 28])
print(df.groupby('age')['salary'].mean())

# 数据清洗
df.dropna()  # 删除缺失值
df.fillna(0)  # 填充缺失值
```

## 3.4 Matplotlib可视化

```python
import matplotlib.pyplot as plt

# 折线图
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('示例图表')
plt.show()

# 柱状图
categories = ['A', 'B', 'C']
values = [10, 20, 15]
plt.bar(categories, values)
plt.show()
```

## 3.5 实战：数据分析项目

```python
import pandas as pd
import matplotlib.pyplot as plt

# 学生成绩分析系统
class GradeAnalyzer:
    def __init__(self):
        self.df = pd.DataFrame({
            'student': ['张三', '李四', '王五', '赵六'],
            'math': [85, 92, 78, 88],
            'english': [90, 85, 82, 91],
            'chinese': [88, 90, 85, 87]
        })
    
    def calculate_stats(self):
        print("各科平均分:")
        print(self.df[['math', 'english', 'chinese']].mean())
        
        print("\n总分排名:")
        self.df['total'] = self.df[['math', 'english', 'chinese']].sum(axis=1)
        print(self.df.sort_values('total', ascending=False))
    
    def visualize(self):
        self.df.plot(x='student', y=['math', 'english', 'chinese'], 
                    kind='bar', figsize=(10, 6))
        plt.title('学生成绩对比')
        plt.ylabel('分数')
        plt.show()

# 使用
analyzer = GradeAnalyzer()
analyzer.calculate_stats()
analyzer.visualize()
```

## 📚 本章小结
- ✅ Python基础语法
- ✅ NumPy数组操作
- ✅ Pandas数据处理
- ✅ Matplotlib可视化
- ✅ 完成数据分析项目

[⬅️ 上一章](./02-机器学习入门.md) | [返回目录](../README.md) | [下一章 ➡️](./04-数学基础.md)
