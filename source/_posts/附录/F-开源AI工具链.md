# 附录F：开源AI工具链大全

## 📚 目录
- [深度学习框架](#深度学习框架)
- [LLM大语言模型](#llm大语言模型)
- [计算机视觉](#计算机视觉)
- [自然语言处理](#自然语言处理)
- [数据处理工具](#数据处理工具)
- [模型部署](#模型部署)
- [MLOps工具](#mlops工具)
- [AutoML自动机器学习](#automl自动机器学习)
- [可视化工具](#可视化工具)
- [开发环境](#开发环境)

---

## 🧠 深度学习框架

### PyTorch
- **GitHub**: https://github.com/pytorch/pytorch
- **Stars**: 75k+
- **特点**: 动态计算图、研究友好
- **安装**: `pip install torch torchvision torchaudio`
- **用途**: 深度学习模型开发

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### TensorFlow
- **GitHub**: https://github.com/tensorflow/tensorflow
- **Stars**: 180k+
- **特点**: 生产部署强大、完整生态
- **安装**: `pip install tensorflow`
- **用途**: 端到端机器学习平台

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### JAX
- **GitHub**: https://github.com/google/jax
- **Stars**: 28k+
- **特点**: 自动微分、JIT编译、GPU/TPU加速
- **安装**: `pip install jax jaxlib`
- **用途**: 高性能数值计算

### PaddlePaddle (飞桨)
- **GitHub**: https://github.com/PaddlePaddle/Paddle
- **Stars**: 21k+
- **特点**: 百度开源、中文友好
- **安装**: `pip install paddlepaddle`
- **用途**: 工业级深度学习平台

### MXNet
- **GitHub**: https://github.com/apache/incubator-mxnet
- **Stars**: 20k+
- **特点**: 高效、灵活、可扩展
- **安装**: `pip install mxnet`

---

## 🤖 LLM大语言模型

### Transformers (Hugging Face)
- **GitHub**: https://github.com/huggingface/transformers
- **Stars**: 120k+
- **特点**: 预训练模型库、易用API
- **安装**: `pip install transformers`
- **模型数量**: 10万+

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
```

### LangChain
- **GitHub**: https://github.com/langchain-ai/langchain
- **Stars**: 80k+
- **特点**: LLM应用开发框架
- **安装**: `pip install langchain`
- **用途**: 构建LLM应用(RAG、Agent)

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt)
```

### LlamaIndex (GPT Index)
- **GitHub**: https://github.com/run-llama/llama_index
- **Stars**: 30k+
- **特点**: 数据框架连接LLM
- **安装**: `pip install llama-index`
- **用途**: 知识库检索、RAG

### vLLM
- **GitHub**: https://github.com/vllm-project/vllm
- **Stars**: 20k+
- **特点**: 高性能LLM推理
- **安装**: `pip install vllm`
- **用途**: 快速LLM部署

### Ollama
- **GitHub**: https://github.com/ollama/ollama
- **Stars**: 60k+
- **特点**: 本地运行LLM
- **安装**: 下载安装包
- **用途**: 本地部署Llama、Mistral等

```bash
ollama run llama2
ollama run mistral
```

### LocalGPT
- **GitHub**: https://github.com/PromtEngineer/localGPT
- **Stars**: 19k+
- **特点**: 本地私有GPT
- **用途**: 企业私有化部署

### Axolotl
- **GitHub**: https://github.com/OpenAccess-AI-Collective/axolotl
- **Stars**: 6k+
- **特点**: LLM微调工具
- **用途**: 模型训练和微调

---

## 👁️ 计算机视觉

### OpenCV
- **GitHub**: https://github.com/opencv/opencv
- **Stars**: 75k+
- **特点**: 经典CV库
- **安装**: `pip install opencv-python`
- **用途**: 图像处理、计算机视觉

```python
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### Ultralytics YOLOv8
- **GitHub**: https://github.com/ultralytics/ultralytics
- **Stars**: 20k+
- **特点**: 最新YOLO目标检测
- **安装**: `pip install ultralytics`
- **用途**: 目标检测、分割、分类

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('image.jpg')
```

### Detectron2
- **GitHub**: https://github.com/facebookresearch/detectron2
- **Stars**: 28k+
- **特点**: Facebook目标检测库
- **安装**: 见官方文档
- **用途**: 目标检测、实例分割

### MMDetection
- **GitHub**: https://github.com/open-mmlab/mmdetection
- **Stars**: 27k+
- **特点**: OpenMMLab检测工具箱
- **安装**: `pip install mmdet`
- **用途**: 目标检测、实例分割

### Segment Anything (SAM)
- **GitHub**: https://github.com/facebookresearch/segment-anything
- **Stars**: 44k+
- **特点**: Meta的图像分割模型
- **安装**: `pip install segment-anything`
- **用途**: 零样本图像分割

### MediaPipe
- **GitHub**: https://github.com/google/mediapipe
- **Stars**: 25k+
- **特点**: Google ML解决方案
- **安装**: `pip install mediapipe`
- **用途**: 人脸检测、手势识别、姿态估计

### EasyOCR
- **GitHub**: https://github.com/JaidedAI/EasyOCR
- **Stars**: 21k+
- **特点**: 简单易用的OCR
- **安装**: `pip install easyocr`
- **支持语言**: 80+

```python
import easyocr

reader = easyocr.Reader(['ch_sim','en'])
result = reader.readtext('image.jpg')
```

### PaddleOCR
- **GitHub**: https://github.com/PaddlePaddle/PaddleOCR
- **Stars**: 38k+
- **特点**: 百度OCR工具
- **安装**: `pip install paddleocr`
- **支持语言**: 80+

---

## 📝 自然语言处理

### spaCy
- **GitHub**: https://github.com/explosion/spaCy
- **Stars**: 28k+
- **特点**: 工业级NLP
- **安装**: `pip install spacy`
- **用途**: 分词、NER、依存分析

```python
import spacy

nlp = spacy.load("zh_core_web_sm")
doc = nlp("我爱北京天安门")
```

### NLTK
- **GitHub**: https://github.com/nltk/nltk
- **Stars**: 13k+
- **特点**: 经典NLP工具包
- **安装**: `pip install nltk`
- **用途**: 文本处理、语言学研究

### Jieba (结巴分词)
- **GitHub**: https://github.com/fxsjy/jieba
- **Stars**: 32k+
- **特点**: 中文分词
- **安装**: `pip install jieba`

```python
import jieba

words = jieba.cut("我来到北京清华大学")
print('/'.join(words))
```

### TextBlob
- **GitHub**: https://github.com/sloria/TextBlob
- **Stars**: 9k+
- **特点**: 简单的NLP
- **安装**: `pip install textblob`
- **用途**: 情感分析、翻译

### Gensim
- **GitHub**: https://github.com/RaRe-Technologies/gensim
- **Stars**: 15k+
- **特点**: 主题建模
- **安装**: `pip install gensim`
- **用途**: Word2Vec、Doc2Vec

---

## 🎨 图像生成

### Stable Diffusion Web UI
- **GitHub**: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- **Stars**: 130k+
- **特点**: SD最流行的Web界面
- **用途**: 文生图、图生图

### ComfyUI
- **GitHub**: https://github.com/comfyanonymous/ComfyUI
- **Stars**: 35k+
- **特点**: 节点式SD界面
- **用途**: 可视化AI绘画工作流

### Diffusers
- **GitHub**: https://github.com/huggingface/diffusers
- **Stars**: 22k+
- **特点**: HF扩散模型库
- **安装**: `pip install diffusers`

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("a cat").images[0]
```

### ControlNet
- **GitHub**: https://github.com/lllyasviel/ControlNet
- **Stars**: 28k+
- **特点**: 精确控制图像生成
- **用途**: 条件图像生成

---

## 📊 数据处理工具

### Pandas
- **GitHub**: https://github.com/pandas-dev/pandas
- **Stars**: 41k+
- **特点**: 数据分析利器
- **安装**: `pip install pandas`

### Polars
- **GitHub**: https://github.com/pola-rs/polars
- **Stars**: 25k+
- **特点**: 高性能DataFrame
- **安装**: `pip install polars`
- **速度**: 比Pandas快10-100倍

### Dask
- **GitHub**: https://github.com/dask/dask
- **Stars**: 12k+
- **特点**: 并行计算
- **安装**: `pip install dask`
- **用途**: 大规模数据处理

### Apache Spark (PySpark)
- **GitHub**: https://github.com/apache/spark
- **Stars**: 38k+
- **特点**: 分布式计算
- **安装**: `pip install pyspark`

---

## 🚀 模型部署

### ONNX Runtime
- **GitHub**: https://github.com/microsoft/onnxruntime
- **Stars**: 12k+
- **特点**: 跨平台推理
- **安装**: `pip install onnxruntime`
- **用途**: 模型优化和部署

### TensorRT
- **GitHub**: https://github.com/NVIDIA/TensorRT
- **Stars**: 9k+
- **特点**: NVIDIA高性能推理
- **用途**: GPU加速推理

### OpenVINO
- **GitHub**: https://github.com/openvinotoolkit/openvino
- **Stars**: 6k+
- **特点**: Intel优化工具
- **用途**: CPU/GPU推理优化

### TorchServe
- **GitHub**: https://github.com/pytorch/serve
- **Stars**: 4k+
- **特点**: PyTorch模型服务
- **安装**: `pip install torchserve`

### BentoML
- **GitHub**: https://github.com/bentoml/BentoML
- **Stars**: 6k+
- **特点**: ML模型服务框架
- **安装**: `pip install bentoml`

```python
import bentoml

@bentoml.service
class MyService:
    @bentoml.api
    def predict(self, input_data):
        return model.predict(input_data)
```

### Triton Inference Server
- **GitHub**: https://github.com/triton-inference-server/server
- **Stars**: 7k+
- **特点**: NVIDIA推理服务器
- **用途**: 多框架模型部署

---

## 🔧 MLOps工具

### MLflow
- **GitHub**: https://github.com/mlflow/mlflow
- **Stars**: 17k+
- **特点**: ML生命周期管理
- **安装**: `pip install mlflow`
- **功能**: 实验跟踪、模型管理

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_metric("rmse", 0.789)
    mlflow.sklearn.log_model(model, "model")
```

### Weights & Biases (wandb)
- **GitHub**: https://github.com/wandb/wandb
- **Stars**: 8k+
- **特点**: 实验跟踪和可视化
- **安装**: `pip install wandb`

```python
import wandb

wandb.init(project="my-project")
wandb.log({"loss": 0.5, "accuracy": 0.95})
```

### DVC (Data Version Control)
- **GitHub**: https://github.com/iterative/dvc
- **Stars**: 13k+
- **特点**: 数据和模型版本控制
- **安装**: `pip install dvc`

```bash
dvc add data/dataset.csv
dvc push
```

### Kedro
- **GitHub**: https://github.com/kedro-org/kedro
- **Stars**: 9k+
- **特点**: 数据科学项目框架
- **安装**: `pip install kedro`

### Airflow
- **GitHub**: https://github.com/apache/airflow
- **Stars**: 34k+
- **特点**: 工作流调度
- **安装**: `pip install apache-airflow`
- **用途**: ML Pipeline编排

### Kubeflow
- **GitHub**: https://github.com/kubeflow/kubeflow
- **Stars**: 14k+
- **特点**: Kubernetes上的ML
- **用途**: 分布式ML训练和部署

---

## 🤖 AutoML自动机器学习

### Auto-sklearn
- **GitHub**: https://github.com/automl/auto-sklearn
- **Stars**: 7k+
- **特点**: 自动ML
- **安装**: `pip install auto-sklearn`

### AutoGluon
- **GitHub**: https://github.com/autogluon/autogluon
- **Stars**: 7k+
- **特点**: AWS AutoML
- **安装**: `pip install autogluon`

### H2O AutoML
- **GitHub**: https://github.com/h2oai/h2o-3
- **Stars**: 6k+
- **特点**: 开源AutoML
- **安装**: `pip install h2o`

### PyCaret
- **GitHub**: https://github.com/pycaret/pycaret
- **Stars**: 8k+
- **特点**: 低代码ML
- **安装**: `pip install pycaret`

```python
from pycaret.classification import *

clf = setup(data, target='target')
best_model = compare_models()
```

### FLAML
- **GitHub**: https://github.com/microsoft/FLAML
- **Stars**: 3k+
- **特点**: 微软AutoML
- **安装**: `pip install flaml`

---

## 📈 可视化工具

### Matplotlib
- **GitHub**: https://github.com/matplotlib/matplotlib
- **Stars**: 19k+
- **安装**: `pip install matplotlib`

### Plotly
- **GitHub**: https://github.com/plotly/plotly.py
- **Stars**: 15k+
- **特点**: 交互式图表
- **安装**: `pip install plotly`

### Seaborn
- **GitHub**: https://github.com/mwaskom/seaborn
- **Stars**: 12k+
- **特点**: 统计可视化
- **安装**: `pip install seaborn`

### Altair
- **GitHub**: https://github.com/altair-viz/altair
- **Stars**: 9k+
- **特点**: 声明式可视化
- **安装**: `pip install altair`

### Gradio
- **GitHub**: https://github.com/gradio-app/gradio
- **Stars**: 28k+
- **特点**: 快速构建ML Demo
- **安装**: `pip install gradio`

```python
import gradio as gr

def predict(image):
    return model(image)

gr.Interface(fn=predict, inputs="image", outputs="label").launch()
```

### Streamlit
- **GitHub**: https://github.com/streamlit/streamlit
- **Stars**: 31k+
- **特点**: ML应用开发框架
- **安装**: `pip install streamlit`

```python
import streamlit as st

st.title("My ML App")
uploaded_file = st.file_uploader("Choose a file")
```

---

## 💻 开发环境

### Jupyter Notebook
- **GitHub**: https://github.com/jupyter/notebook
- **Stars**: 11k+
- **安装**: `pip install notebook`

### JupyterLab
- **GitHub**: https://github.com/jupyterlab/jupyterlab
- **Stars**: 14k+
- **特点**: 下一代Jupyter
- **安装**: `pip install jupyterlab`

### Google Colab
- **网址**: https://colab.research.google.com/
- **特点**: 免费GPU
- **用途**: 在线开发和训练

### Kaggle Notebooks
- **网址**: https://www.kaggle.com/code
- **特点**: 免费GPU/TPU
- **用途**: 竞赛和学习

---

## 🎯 专用工具

### 语音处理

#### Whisper (OpenAI)
- **GitHub**: https://github.com/openai/whisper
- **Stars**: 60k+
- **特点**: 语音识别
- **安装**: `pip install openai-whisper`

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
```

#### ESPnet
- **GitHub**: https://github.com/espnet/espnet
- **Stars**: 7k+
- **特点**: 端到端语音处理
- **用途**: ASR、TTS

#### Coqui TTS
- **GitHub**: https://github.com/coqui-ai/TTS
- **Stars**: 30k+
- **特点**: 文本转语音
- **安装**: `pip install TTS`

### 推荐系统

#### RecBole
- **GitHub**: https://github.com/RUCAIBox/RecBole
- **Stars**: 3k+
- **特点**: 推荐系统库
- **安装**: `pip install recbole`

#### LightFM
- **GitHub**: https://github.com/lyst/lightfm
- **Stars**: 4k+
- **特点**: 混合推荐
- **安装**: `pip install lightfm`

### 图神经网络

#### PyTorch Geometric
- **GitHub**: https://github.com/pyg-team/pytorch_geometric
- **Stars**: 20k+
- **特点**: 图神经网络
- **安装**: `pip install torch-geometric`

#### DGL (Deep Graph Library)
- **GitHub**: https://github.com/dmlc/dgl
- **Stars**: 13k+
- **特点**: 图深度学习
- **安装**: `pip install dgl`

### 强化学习

#### Stable-Baselines3
- **GitHub**: https://github.com/DLR-RM/stable-baselines3
- **Stars**: 7k+
- **特点**: 强化学习算法
- **安装**: `pip install stable-baselines3`

#### OpenAI Gym
- **GitHub**: https://github.com/openai/gym
- **Stars**: 34k+
- **特点**: 强化学习环境
- **安装**: `pip install gym`

#### Ray RLlib
- **GitHub**: https://github.com/ray-project/ray
- **Stars**: 31k+
- **特点**: 分布式强化学习
- **安装**: `pip install ray[rllib]`

---

## 🔍 模型压缩与优化

### ONNX
- **GitHub**: https://github.com/onnx/onnx
- **Stars**: 17k+
- **特点**: 模型交换格式
- **安装**: `pip install onnx`

### TensorFlow Lite
- **文档**: https://www.tensorflow.org/lite
- **特点**: 移动端部署
- **用途**: 边缘设备推理

### PyTorch Mobile
- **文档**: https://pytorch.org/mobile
- **特点**: 移动端PyTorch
- **用途**: iOS/Android部署

### Neural Compressor
- **GitHub**: https://github.com/intel/neural-compressor
- **Stars**: 2k+
- **特点**: 模型压缩
- **安装**: `pip install neural-compressor`

---

## 📦 一键安装脚本

### Python AI开发环境

```bash
# 基础科学计算
pip install numpy pandas scipy matplotlib seaborn

# 机器学习
pip install scikit-learn xgboost lightgbm

# 深度学习
pip install torch torchvision torchaudio
pip install tensorflow

# NLP
pip install transformers datasets tokenizers
pip install spacy jieba

# CV
pip install opencv-python pillow
pip install ultralytics

# LLM应用
pip install langchain openai
pip install chromadb

# 可视化
pip install plotly gradio streamlit

# 开发工具
pip install jupyter jupyterlab
pip install mlflow wandb

# 部署
pip install fastapi uvicorn
pip install bentoml
```

### Conda环境配置

```bash
# 创建环境
conda create -n ai-tools python=3.10 -y
conda activate ai-tools

# 安装核心库
conda install numpy pandas matplotlib seaborn scikit-learn -y
conda install pytorch torchvision torchaudio -c pytorch -y

# 使用pip安装其他工具
pip install transformers langchain gradio streamlit
```

---

## 🌟 推荐组合

### 组合1: LLM应用开发
```
OpenAI API / Ollama (模型)
+ LangChain (框架)
+ ChromaDB (向量数据库)
+ Gradio (界面)
+ FastAPI (API)
```

### 组合2: 计算机视觉
```
PyTorch (框架)
+ Ultralytics (YOLO)
+ OpenCV (图像处理)
+ Gradio (Demo)
+ ONNX (部署)
```

### 组合3: 数据科学
```
Pandas (数据处理)
+ scikit-learn (机器学习)
+ Matplotlib/Plotly (可视化)
+ Jupyter (开发环境)
+ MLflow (实验管理)
```

---

## 📚 学习资源

### GitHub精选
- **Awesome Machine Learning**: https://github.com/josephmisiti/awesome-machine-learning
- **Awesome Deep Learning**: https://github.com/ChristosChristofidis/awesome-deep-learning
- **Awesome Computer Vision**: https://github.com/jbhuang0604/awesome-computer-vision

### Papers With Code
- **网址**: https://paperswithcode.com/
- **特点**: 论文+代码实现

---

## 🔗 快速链接

- [返回目录](../README.md)
- [环境配置](./A-环境配置.md)
- [学习资源](./C-学习资源.md)
- [工具速查](./D-工具速查.md)

---

**最后更新**: 2025-10-18  
**收录工具**: 100+  
**分类数量**: 12个主要类别
