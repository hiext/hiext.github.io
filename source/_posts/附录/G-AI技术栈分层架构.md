# 附录G：AI技术栈分层架构

> 从底层硬件到上层应用的完整AI技术栈体系

## 📊 架构总览

```
┌─────────────────────────────────────────────────┐
│          第7层：应用层 (Applications)            │
│        商业应用、垂直领域解决方案                  │
├─────────────────────────────────────────────────┤
│       第6层：开发工具层 (Dev Tools)              │
│      IDE、调试工具、可视化界面、低代码平台          │
├─────────────────────────────────────────────────┤
│        第5层：MLOps层 (MLOps)                   │
│   实验管理、模型版本、CI/CD、监控告警              │
├─────────────────────────────────────────────────┤
│       第4层：模型层 (Models & Algorithms)        │
│    预训练模型、算法库、AutoML                     │
├─────────────────────────────────────────────────┤
│        第3层：框架层 (Frameworks)                │
│   深度学习框架、数据处理框架                      │
├─────────────────────────────────────────────────┤
│      第2层：运行时层 (Runtime & Deploy)          │
│   推理引擎、模型优化、容器化、API服务              │
├─────────────────────────────────────────────────┤
│       第1层：基础设施层 (Infrastructure)          │
│     硬件加速、驱动、计算资源、存储                │
└─────────────────────────────────────────────────┘
```

---

## 🔧 第1层：基础设施层 (Infrastructure)

> 提供计算、存储、网络等底层资源

### 1.1 硬件加速器

#### NVIDIA GPU
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **cuDNN**: 深度学习加速库
- **GPU型号**: 
  - 入门: RTX 3060 (12GB)
  - 进阶: RTX 4090 (24GB)
  - 专业: A100 (40GB/80GB), H100

```bash
# 检查GPU
nvidia-smi

# 检查CUDA版本
nvcc --version
```

#### AMD GPU
- **ROCm**: https://github.com/RadeonOpenCompute/ROCm
- **GPU型号**: MI100, MI250

#### Google TPU
- **Cloud TPU**: https://cloud.google.com/tpu
- **用途**: 大规模训练

#### Intel 加速器
- **oneAPI**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html
- **Habana Gaudi**: AI训练处理器

### 1.2 驱动与底层库

#### CUDA 生态
```bash
# CUDA 11.8 安装
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

#### cuBLAS / cuDNN
- **cuBLAS**: 线性代数加速
- **cuDNN**: 深度学习原语加速
- **NCCL**: 多GPU通信

### 1.3 云计算平台

#### 主流云平台
| 平台 | GPU类型 | 特点 |
|------|---------|------|
| **AWS** | A100, V100 | SageMaker |
| **Google Cloud** | TPU, A100 | Vertex AI |
| **Azure** | A100, V100 | Azure ML |
| **阿里云** | V100, A100 | PAI平台 |
| **腾讯云** | V100, T4 | TI平台 |

#### GPU云服务
- **Lambda Labs**: https://lambdalabs.com/
- **Vast.ai**: https://vast.ai/
- **RunPod**: https://www.runpod.io/

### 1.4 存储系统

#### 对象存储
- **AWS S3**: 云存储标准
- **MinIO**: https://github.com/minio/minio (开源S3)
- **阿里云OSS**: 对象存储

#### 数据湖
- **Apache Hadoop HDFS**: 分布式文件系统
- **Delta Lake**: https://github.com/delta-io/delta

---

## ⚙️ 第2层：运行时与部署层 (Runtime & Deploy)

> 模型推理、优化、容器化、服务化

### 2.1 推理引擎

#### ONNX Runtime
- **GitHub**: https://github.com/microsoft/onnxruntime
- **Stars**: 12k+
- **安装**: `pip install onnxruntime-gpu`

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_data})
```

#### TensorRT
- **GitHub**: https://github.com/NVIDIA/TensorRT
- **特点**: NVIDIA高性能推理
- **加速**: 5-10倍

```python
import tensorrt as trt

# 构建引擎
builder = trt.Builder(logger)
engine = builder.build_cuda_engine(network)
```

#### OpenVINO
- **GitHub**: https://github.com/openvinotoolkit/openvino
- **特点**: Intel优化工具
- **支持**: CPU, GPU, VPU

#### TFLite / CoreML
- **TensorFlow Lite**: 移动端推理
- **Core ML**: iOS设备推理

### 2.2 模型优化

#### 量化工具
- **ONNX Quantization**: `pip install onnxruntime-tools`
- **PyTorch Quantization**: 内置
- **TensorFlow Model Optimization**: `pip install tensorflow-model-optimization`

```python
# PyTorch动态量化
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### 模型压缩
- **Neural Compressor**: https://github.com/intel/neural-compressor
- **Distiller**: https://github.com/IntelLabs/distiller

### 2.3 容器化

#### Docker
- **官网**: https://www.docker.com/
- **NVIDIA Container Toolkit**: GPU容器支持

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN pip install torch torchvision

COPY app.py /app/
CMD ["python", "/app/app.py"]
```

#### Kubernetes
- **K8s**: 容器编排
- **KubeFlow**: ML on K8s

### 2.4 API服务框架

#### FastAPI
- **GitHub**: https://github.com/tiangolo/fastapi
- **Stars**: 70k+
- **安装**: `pip install fastapi uvicorn`

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    result = model.predict(data)
    return {"result": result}
```

#### Flask
- **GitHub**: https://github.com/pallets/flask
- **特点**: 轻量级Web框架

#### TorchServe
- **GitHub**: https://github.com/pytorch/serve
- **特点**: PyTorch官方服务框架

#### TensorFlow Serving
- **GitHub**: https://github.com/tensorflow/serving
- **特点**: TF官方服务

#### BentoML
- **GitHub**: https://github.com/bentoml/BentoML
- **Stars**: 6k+
- **特点**: 统一ML服务框架

### 2.5 边缘部署

#### ONNX.js
- **浏览器端推理**: https://github.com/microsoft/onnxjs

#### TensorFlow.js
- **JavaScript ML**: https://www.tensorflow.org/js

---

## 🧠 第3层：框架层 (Frameworks)

> 深度学习、数据处理、计算框架

### 3.1 深度学习框架

#### PyTorch
- **GitHub**: https://github.com/pytorch/pytorch
- **Stars**: 75k+
- **生态**: 研究首选

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters())
```

#### TensorFlow / Keras
- **GitHub**: https://github.com/tensorflow/tensorflow
- **Stars**: 180k+
- **生态**: 生产首选

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

#### JAX
- **GitHub**: https://github.com/google/jax
- **Stars**: 28k+
- **特点**: 高性能数值计算

#### PaddlePaddle
- **GitHub**: https://github.com/PaddlePaddle/Paddle
- **Stars**: 21k+
- **特点**: 百度开源，中文友好

#### MXNet
- **GitHub**: https://github.com/apache/incubator-mxnet
- **特点**: AWS支持

### 3.2 数据处理框架

#### NumPy
- **GitHub**: https://github.com/numpy/numpy
- **特点**: 科学计算基础

#### Pandas
- **GitHub**: https://github.com/pandas-dev/pandas
- **Stars**: 41k+
- **特点**: 数据分析核心

```python
import pandas as pd

df = pd.read_csv('data.csv')
df_clean = df.dropna().groupby('category').mean()
```

#### Polars
- **GitHub**: https://github.com/pola-rs/polars
- **Stars**: 25k+
- **特点**: 比Pandas快10-100倍

#### Dask
- **GitHub**: https://github.com/dask/dask
- **特点**: 并行计算、大数据

#### Apache Spark (PySpark)
- **GitHub**: https://github.com/apache/spark
- **特点**: 分布式大数据处理

### 3.3 图像处理

#### OpenCV
- **GitHub**: https://github.com/opencv/opencv
- **Stars**: 75k+

#### Pillow (PIL)
- **GitHub**: https://github.com/python-pillow/Pillow
- **特点**: Python图像库

#### Albumentations
- **GitHub**: https://github.com/albumentations-team/albumentations
- **特点**: 图像增强

### 3.4 NLP工具

#### spaCy
- **GitHub**: https://github.com/explosion/spaCy
- **Stars**: 28k+

#### NLTK
- **GitHub**: https://github.com/nltk/nltk
- **Stars**: 13k+

#### Jieba
- **GitHub**: https://github.com/fxsjy/jieba
- **Stars**: 32k+

---

## 🤖 第4层：模型与算法层 (Models & Algorithms)

> 预训练模型、算法实现、AutoML

### 4.1 预训练模型库

#### Hugging Face Transformers
- **GitHub**: https://github.com/huggingface/transformers
- **Stars**: 120k+
- **模型数**: 10万+

```python
from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# 文本生成
generator = pipeline("text-generation", model="gpt2")
text = generator("Once upon a time")
```

#### Hugging Face Hub
- **网址**: https://huggingface.co/models
- **功能**: 模型托管、下载、分享

#### TensorFlow Hub
- **网址**: https://tfhub.dev/
- **特点**: TF预训练模型

#### PyTorch Hub
- **网址**: https://pytorch.org/hub/
- **特点**: PyTorch模型库

#### Model Zoo
- **ONNX Model Zoo**: https://github.com/onnx/models
- **MMPretrain**: https://github.com/open-mmlab/mmpretrain

### 4.2 计算机视觉模型

#### Ultralytics YOLOv8
- **GitHub**: https://github.com/ultralytics/ultralytics
- **用途**: 目标检测

#### MMDetection
- **GitHub**: https://github.com/open-mmlab/mmdetection
- **用途**: 目标检测工具箱

#### Detectron2
- **GitHub**: https://github.com/facebookresearch/detectron2
- **用途**: Facebook检测库

#### Segment Anything (SAM)
- **GitHub**: https://github.com/facebookresearch/segment-anything
- **用途**: 图像分割

### 4.3 NLP模型

#### 中文模型
- **ChatGLM**: https://github.com/THUDM/ChatGLM-6B
- **Qwen**: https://github.com/QwenLM/Qwen
- **Baichuan**: https://github.com/baichuan-inc/Baichuan-7B

#### 英文模型
- **Llama 2**: Meta开源LLM
- **Mistral**: https://mistral.ai/
- **GPT-2**: OpenAI开源模型

### 4.4 多模态模型

#### CLIP
- **GitHub**: https://github.com/openai/CLIP
- **用途**: 图文对齐

#### Stable Diffusion
- **GitHub**: https://github.com/Stability-AI/stablediffusion
- **用途**: 文生图

#### Whisper
- **GitHub**: https://github.com/openai/whisper
- **Stars**: 60k+
- **用途**: 语音识别

### 4.5 AutoML工具

#### Auto-sklearn
- **GitHub**: https://github.com/automl/auto-sklearn
- **安装**: `pip install auto-sklearn`

#### AutoGluon
- **GitHub**: https://github.com/autogluon/autogluon
- **特点**: AWS AutoML

#### PyCaret
- **GitHub**: https://github.com/pycaret/pycaret
- **特点**: 低代码ML

```python
from pycaret.classification import *

clf = setup(data, target='target')
best_model = compare_models()
```

#### H2O AutoML
- **GitHub**: https://github.com/h2oai/h2o-3

---

## 🔧 第5层：MLOps层 (MLOps & Lifecycle)

> 实验管理、版本控制、CI/CD、监控

### 5.1 实验跟踪

#### MLflow
- **GitHub**: https://github.com/mlflow/mlflow
- **Stars**: 17k+
- **功能**: 实验跟踪、模型注册

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_metric("rmse", 0.789)
    mlflow.sklearn.log_model(model, "model")
```

#### Weights & Biases
- **GitHub**: https://github.com/wandb/wandb
- **Stars**: 8k+
- **特点**: 可视化强大

```python
import wandb

wandb.init(project="my-project")
wandb.log({"loss": 0.5, "accuracy": 0.95})
```

#### TensorBoard
- **集成**: TensorFlow内置
- **特点**: 训练可视化

#### Aim
- **GitHub**: https://github.com/aimhubio/aim
- **特点**: 开源实验跟踪

### 5.2 版本控制

#### DVC (Data Version Control)
- **GitHub**: https://github.com/iterative/dvc
- **Stars**: 13k+
- **用途**: 数据和模型版本控制

```bash
dvc init
dvc add data/dataset.csv
dvc push

# 切换版本
git checkout v1.0
dvc checkout
```

#### Git LFS
- **用途**: 大文件版本控制

#### Pachyderm
- **GitHub**: https://github.com/pachyderm/pachyderm
- **特点**: 数据版本化和管道

### 5.3 工作流编排

#### Apache Airflow
- **GitHub**: https://github.com/apache/airflow
- **Stars**: 34k+
- **用途**: 工作流调度

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('ml_pipeline', schedule_interval='@daily')

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)
```

#### Prefect
- **GitHub**: https://github.com/PrefectHQ/prefect
- **特点**: 现代工作流引擎

#### Kubeflow
- **GitHub**: https://github.com/kubeflow/kubeflow
- **特点**: K8s上的ML

#### Kedro
- **GitHub**: https://github.com/kedro-org/kedro
- **特点**: 数据科学项目框架

### 5.4 模型注册与管理

#### MLflow Model Registry
- **功能**: 模型版本管理
- **生命周期**: Staging → Production

#### ModelDB
- **GitHub**: https://github.com/VertaAI/modeldb

### 5.5 监控与告警

#### Prometheus
- **GitHub**: https://github.com/prometheus/prometheus
- **用途**: 指标监控

#### Grafana
- **网址**: https://grafana.com/
- **用途**: 可视化监控

#### Evidently AI
- **GitHub**: https://github.com/evidentlyai/evidently
- **用途**: ML模型监控

---

## 🛠️ 第6层：开发工具层 (Development Tools)

> IDE、Notebook、可视化、低代码

### 6.1 IDE与编辑器

#### Jupyter Notebook / Lab
- **GitHub**: https://github.com/jupyter/notebook
- **安装**: `pip install jupyterlab`

```bash
jupyter lab
```

#### VS Code
- **扩展**: 
  - Python
  - Jupyter
  - Pylance

#### PyCharm
- **版本**: Professional (推荐)
- **特点**: 强大的Python IDE

#### Google Colab
- **网址**: https://colab.research.google.com/
- **特点**: 免费GPU

#### Kaggle Notebooks
- **网址**: https://www.kaggle.com/code
- **特点**: 免费GPU/TPU

### 6.2 可视化与UI

#### Gradio
- **GitHub**: https://github.com/gradio-app/gradio
- **Stars**: 28k+
- **用途**: 快速构建ML Demo

```python
import gradio as gr

def predict(image):
    return model(image)

gr.Interface(fn=predict, inputs="image", outputs="label").launch()
```

#### Streamlit
- **GitHub**: https://github.com/streamlit/streamlit
- **Stars**: 31k+
- **用途**: ML应用开发

```python
import streamlit as st

st.title("My ML App")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file:
    result = model.predict(uploaded_file)
    st.write(result)
```

#### Plotly Dash
- **GitHub**: https://github.com/plotly/dash
- **用途**: 分析应用

#### Panel
- **GitHub**: https://github.com/holoviz/panel
- **特点**: 灵活的仪表板

### 6.3 低代码/无代码平台

#### Hugging Face Spaces
- **网址**: https://huggingface.co/spaces
- **特点**: 托管Gradio/Streamlit应用

#### Google Teachable Machine
- **网址**: https://teachablemachine.withgoogle.com/
- **特点**: 无代码训练

#### Microsoft Lobe
- **网址**: https://www.lobe.ai/
- **特点**: 图形化训练

### 6.4 调试与分析

#### TensorBoard
- **用途**: 训练可视化

#### Netron
- **GitHub**: https://github.com/lutzroeder/netron
- **用途**: 模型可视化

#### PyTorch Profiler
- **用途**: 性能分析

---

## 🚀 第7层：应用层 (Applications)

> 垂直领域解决方案、商业应用

### 7.1 LLM应用框架

#### LangChain
- **GitHub**: https://github.com/langchain-ai/langchain
- **Stars**: 80k+
- **用途**: LLM应用开发

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI()
prompt = PromptTemplate(
    input_variables=["product"],
    template="为{product}写一段营销文案"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("AI助手")
```

#### LlamaIndex
- **GitHub**: https://github.com/run-llama/llama_index
- **用途**: 知识库检索、RAG

#### Semantic Kernel
- **GitHub**: https://github.com/microsoft/semantic-kernel
- **用途**: 微软LLM框架

#### Haystack
- **GitHub**: https://github.com/deepset-ai/haystack
- **用途**: NLP应用框架

### 7.2 向量数据库

#### ChromaDB
- **GitHub**: https://github.com/chroma-core/chroma
- **安装**: `pip install chromadb`

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(
    documents=["这是一段文本"],
    ids=["id1"]
)

results = collection.query(query_texts=["查询"], n_results=1)
```

#### Milvus
- **GitHub**: https://github.com/milvus-io/milvus
- **特点**: 高性能向量数据库

#### Weaviate
- **GitHub**: https://github.com/weaviate/weaviate

#### Pinecone
- **网址**: https://www.pinecone.io/
- **特点**: 托管向量数据库

#### Qdrant
- **GitHub**: https://github.com/qdrant/qdrant

### 7.3 AI绘画工具

#### Stable Diffusion WebUI
- **GitHub**: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- **Stars**: 130k+
- **特点**: 最流行的SD界面

#### ComfyUI
- **GitHub**: https://github.com/comfyanonymous/ComfyUI
- **Stars**: 35k+
- **特点**: 节点式工作流

#### Fooocus
- **GitHub**: https://github.com/lllyasviel/Fooocus
- **特点**: 简化的SD界面

### 7.4 计算机视觉应用

#### OCR工具
- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR (38k stars)
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR (21k stars)
- **Tesseract**: https://github.com/tesseract-ocr/tesseract

#### 人脸识别
- **Face Recognition**: https://github.com/ageitgey/face_recognition
- **DeepFace**: https://github.com/serengil/deepface
- **InsightFace**: https://github.com/deepinsight/insightface

#### 姿态估计
- **OpenPose**: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- **MediaPipe**: https://github.com/google/mediapipe

### 7.5 语音应用

#### 语音识别 (ASR)
- **Whisper**: https://github.com/openai/whisper (60k stars)
- **Vosk**: https://github.com/alphacep/vosk-api
- **DeepSpeech**: https://github.com/mozilla/DeepSpeech

#### 语音合成 (TTS)
- **Coqui TTS**: https://github.com/coqui-ai/TTS (30k stars)
- **Bark**: https://github.com/suno-ai/bark
- **VITS**: https://github.com/jaywalnut310/vits

### 7.6 推荐系统

#### RecBole
- **GitHub**: https://github.com/RUCAIBox/RecBole
- **特点**: 推荐系统库

#### LightFM
- **GitHub**: https://github.com/lyst/lightfm
- **特点**: 混合推荐

#### Surprise
- **GitHub**: https://github.com/NicolasHug/Surprise
- **特点**: scikit风格推荐库

### 7.7 时间序列

#### Prophet
- **GitHub**: https://github.com/facebook/prophet
- **特点**: Facebook时间序列

#### Darts
- **GitHub**: https://github.com/unit8co/darts
- **特点**: 时间序列预测

#### GluonTS
- **GitHub**: https://github.com/awslabs/gluonts
- **特点**: AWS时间序列

### 7.8 强化学习

#### Stable-Baselines3
- **GitHub**: https://github.com/DLR-RM/stable-baselines3
- **安装**: `pip install stable-baselines3`

#### OpenAI Gym
- **GitHub**: https://github.com/openai/gym
- **特点**: RL环境

#### Ray RLlib
- **GitHub**: https://github.com/ray-project/ray
- **特点**: 分布式RL

---

## 🎯 典型技术栈组合

### 组合1：LLM应用全栈

```
┌─ 应用层: LangChain + Gradio
├─ 数据层: ChromaDB (向量数据库)
├─ 模型层: Hugging Face Transformers
├─ 框架层: PyTorch
├─ 部署层: FastAPI + Docker
├─ MLOps: MLflow + Wandb
└─ 基础设施: AWS/阿里云 + NVIDIA GPU
```

**安装命令**:
```bash
pip install langchain transformers chromadb
pip install gradio fastapi uvicorn
pip install mlflow wandb
pip install torch
```

### 组合2：计算机视觉全栈

```
┌─ 应用层: Gradio UI
├─ 模型层: Ultralytics YOLOv8
├─ 框架层: PyTorch + OpenCV
├─ 部署层: ONNX Runtime + FastAPI
├─ MLOps: MLflow + DVC
└─ 基础设施: NVIDIA GPU + TensorRT
```

**安装命令**:
```bash
pip install ultralytics opencv-python
pip install torch torchvision
pip install onnxruntime-gpu
pip install gradio fastapi mlflow dvc
```

### 组合3：数据科学全栈

```
┌─ 应用层: Streamlit
├─ AutoML: PyCaret
├─ 算法层: scikit-learn + XGBoost
├─ 数据层: Pandas + Polars
├─ 可视化: Plotly + Seaborn
├─ MLOps: MLflow + DVC
└─ 基础设施: Jupyter Lab
```

**安装命令**:
```bash
pip install pandas polars
pip install scikit-learn xgboost pycaret
pip install plotly seaborn matplotlib
pip install streamlit jupyterlab mlflow
```

---

## 📊 学习路径建议

### 阶段1：基础设施熟悉 (1周)
1. 配置Python环境
2. 安装CUDA (如有GPU)
3. 熟悉Jupyter Notebook

### 阶段2：框架层掌握 (1-2个月)
1. 学习NumPy、Pandas
2. 掌握PyTorch或TensorFlow
3. 了解OpenCV基础

### 阶段3：模型层应用 (2-3个月)
1. 使用Hugging Face模型
2. 训练自己的模型
3. 尝试AutoML工具

### 阶段4：部署与MLOps (1-2个月)
1. 学习FastAPI
2. 掌握Docker容器化
3. 使用MLflow跟踪实验

### 阶段5：应用层开发 (持续)
1. 开发LLM应用 (LangChain)
2. 构建CV应用 (YOLO)
3. 创建完整项目

---

## 🔗 相关文档

- [返回目录](../README.md)
- [F-开源AI工具链大全](./F-开源AI工具链.md)
- [AI工具链速查卡](../AI工具链速查卡.md)
- [A-环境配置](./A-环境配置.md)

---

**最后更新**: 2025-10-18  
**架构层次**: 7层完整体系  
**覆盖工具**: 150+
