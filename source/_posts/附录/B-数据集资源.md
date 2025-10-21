# 附录B：数据集资源大全

## 📊 综合数据集平台

### 1. Kaggle
- **网址**: https://www.kaggle.com/datasets
- **特点**: 最大的数据科学社区，海量数据集
- **推荐数据集**:
  - Titanic - 入门级分类问题
  - House Prices - 回归问题
  - MNIST - 手写数字识别
  - ImageNet - 大规模图像分类

### 2. UCI Machine Learning Repository
- **网址**: https://archive.ics.uci.edu/ml/
- **特点**: 经典机器学习数据集
- **推荐**:
  - Iris - 鸢尾花分类
  - Wine Quality - 葡萄酒质量预测
  - Adult Income - 收入预测

### 3. Google Dataset Search
- **网址**: https://datasetsearch.research.google.com/
- **特点**: 谷歌的数据集搜索引擎

## 🖼️ 计算机视觉数据集

### 图像分类
- **CIFAR-10/100**: 60000张32x32彩色图像
- **ImageNet**: 1400万张标注图像
- **Fashion-MNIST**: 时尚物品图像
- **COCO**: 33万张图像，80个类别

### 目标检测
- **PASCAL VOC**: 目标检测经典数据集
- **COCO Detection**: 20万标注图像
- **Open Images**: 900万张图像

### 人脸识别
- **LFW**: 人脸识别基准
- **CelebA**: 20万名人人脸
- **VGGFace2**: 330万张人脸图像

## 📝 自然语言处理数据集

### 中文数据集
- **中文维基百科**: https://dumps.wikimedia.org/zhwiki/
- **新闻语料库**: THUCNews、搜狗新闻
- **情感分析**: ChnSentiCorp、外卖评论
- **问答系统**: DuReader、WebQA

### 英文数据集
- **GLUE**: NLP任务基准测试
- **SQuAD**: 问答数据集
- **IMDb**: 电影评论情感分析
- **Amazon Reviews**: 商品评论

## 🎵 音频数据集

- **LibriSpeech**: 1000小时英文语音
- **Common Voice**: Mozilla开源语音
- **Free Music Archive**: 音乐分类
- **UrbanSound8K**: 城市声音分类

## 📈 时间序列数据集

- **股票数据**: Yahoo Finance, Alpha Vantage
- **天气数据**: NOAA气象数据
- **能源消耗**: UCI个人能耗数据
- **交通流量**: Uber Movement

## 🏥 医疗健康数据集

- **MIMIC-III**: 重症监护数据
- **Chest X-Ray**: 胸部X光片
- **Skin Cancer MNIST**: 皮肤癌图像
- **COVID-19**: 新冠肺炎数据集

## 🛒 推荐系统数据集

- **MovieLens**: 电影评分数据
- **Amazon Product**: 商品评论
- **Yelp**: 餐厅评论
- **Netflix Prize**: 电影推荐

## 📱 社交网络数据集

- **Twitter**: 推文数据
- **Reddit**: 帖子和评论
- **Facebook**: 社交网络图

## 🎮 游戏与强化学习

- **OpenAI Gym**: 强化学习环境
- **Atari Games**: 雅达利游戏
- **StarCraft II**: 星际争霸

## 🚗 自动驾驶数据集

- **KITTI**: 自动驾驶基准
- **nuScenes**: 自动驾驶全景数据
- **Waymo Open**: Waymo开放数据集

## 💡 数据集使用技巧

### 1. 数据下载
```python
# Kaggle数据集下载
!pip install kaggle
!kaggle datasets download -d dataset-name

# TensorFlow数据集
import tensorflow_datasets as tfds
dataset = tfds.load('mnist')

# Hugging Face数据集
from datasets import load_dataset
dataset = load_dataset('imdb')
```

### 2. 数据增强
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

### 3. 数据清洗
```python
import pandas as pd

# 处理缺失值
df.dropna()
df.fillna(df.mean())

# 处理异常值
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['column'] < (Q1 - 1.5 * IQR)) | (df['column'] > (Q3 + 1.5 * IQR)))]
```

## 🔗 数据集API

### Hugging Face Datasets
```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('squad')

# 查看信息
print(dataset)
```

### TensorFlow Datasets
```python
import tensorflow_datasets as tfds

# 列出所有数据集
print(tfds.list_builders())

# 加载数据集
dataset = tfds.load('cifar10', split='train')
```

## ⚠️ 数据使用注意事项

1. **版权问题**: 注意数据集的使用许可
2. **隐私保护**: 处理个人数据需谨慎
3. **数据平衡**: 注意类别不平衡问题
4. **数据泄露**: 避免测试集污染
5. **数据质量**: 检查标注错误

## 📚 推荐资源

- **Papers With Code**: https://paperswithcode.com/datasets
- **Awesome Public Datasets**: GitHub仓库
- **Data.gov**: 美国政府开放数据
- **EU Open Data**: 欧盟开放数据

[返回目录](../README.md)
