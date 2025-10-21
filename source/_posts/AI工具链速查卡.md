# 🚀 AI工具链速查卡

## 🔥 最常用工具（新手必备）

### 1️⃣ 深度学习框架
```bash
# PyTorch (推荐)
pip install torch torchvision torchaudio

# TensorFlow
pip install tensorflow
```

### 2️⃣ 数据处理
```bash
pip install pandas numpy matplotlib seaborn
```

### 3️⃣ 机器学习
```bash
pip install scikit-learn xgboost
```

### 4️⃣ LLM应用开发
```bash
pip install transformers langchain openai
pip install chromadb  # 向量数据库
```

### 5️⃣ 计算机视觉
```bash
pip install opencv-python ultralytics
```

### 6️⃣ 快速Demo
```bash
pip install gradio streamlit
```

---

## 💎 分类工具链

### 🧠 深度学习
| 工具 | 用途 | 热度 |
|------|------|------|
| PyTorch | 深度学习框架 | ⭐⭐⭐⭐⭐ |
| TensorFlow | 端到端ML平台 | ⭐⭐⭐⭐⭐ |
| JAX | 高性能计算 | ⭐⭐⭐ |

### 🤖 LLM大模型
| 工具 | 用途 | 热度 |
|------|------|------|
| Transformers | 预训练模型库 | ⭐⭐⭐⭐⭐ |
| LangChain | LLM应用框架 | ⭐⭐⭐⭐⭐ |
| Ollama | 本地运行LLM | ⭐⭐⭐⭐ |
| vLLM | 高性能推理 | ⭐⭐⭐⭐ |

### 👁️ 计算机视觉
| 工具 | 用途 | 热度 |
|------|------|------|
| OpenCV | 图像处理 | ⭐⭐⭐⭐⭐ |
| YOLOv8 | 目标检测 | ⭐⭐⭐⭐⭐ |
| SAM | 图像分割 | ⭐⭐⭐⭐ |
| EasyOCR | 文字识别 | ⭐⭐⭐⭐ |

### 📝 自然语言处理
| 工具 | 用途 | 热度 |
|------|------|------|
| spaCy | 工业级NLP | ⭐⭐⭐⭐⭐ |
| Jieba | 中文分词 | ⭐⭐⭐⭐⭐ |
| NLTK | 经典NLP | ⭐⭐⭐⭐ |

### 🎨 图像生成
| 工具 | 用途 | 热度 |
|------|------|------|
| SD WebUI | Stable Diffusion界面 | ⭐⭐⭐⭐⭐ |
| ComfyUI | 节点式SD | ⭐⭐⭐⭐ |
| Diffusers | HF扩散模型 | ⭐⭐⭐⭐ |

### 🚀 模型部署
| 工具 | 用途 | 热度 |
|------|------|------|
| ONNX Runtime | 跨平台推理 | ⭐⭐⭐⭐⭐ |
| FastAPI | Web API | ⭐⭐⭐⭐⭐ |
| BentoML | ML服务 | ⭐⭐⭐⭐ |
| Docker | 容器化 | ⭐⭐⭐⭐⭐ |

### 🔧 MLOps
| 工具 | 用途 | 热度 |
|------|------|------|
| MLflow | 实验跟踪 | ⭐⭐⭐⭐⭐ |
| Wandb | 可视化 | ⭐⭐⭐⭐ |
| DVC | 数据版本控制 | ⭐⭐⭐⭐ |

### 📊 可视化
| 工具 | 用途 | 热度 |
|------|------|------|
| Gradio | 快速Demo | ⭐⭐⭐⭐⭐ |
| Streamlit | ML应用 | ⭐⭐⭐⭐⭐ |
| Plotly | 交互图表 | ⭐⭐⭐⭐ |

---

## 📦 场景化安装方案

### 场景1: LLM应用开发者
```bash
# 核心工具
pip install transformers langchain openai
pip install chromadb faiss-cpu

# 向量数据库
pip install qdrant-client pinecone-client

# Web界面
pip install gradio streamlit

# API开发
pip install fastapi uvicorn
```

### 场景2: 计算机视觉工程师
```bash
# 核心框架
pip install torch torchvision opencv-python

# 目标检测
pip install ultralytics

# 图像增强
pip install albumentations imgaug

# OCR
pip install easyocr paddleocr

# 可视化
pip install gradio
```

### 场景3: NLP工程师
```bash
# 核心库
pip install transformers datasets tokenizers

# 中文处理
pip install jieba pkuseg

# 工业级
pip install spacy

# 训练框架
pip install pytorch-lightning
```

### 场景4: 数据科学家
```bash
# 数据处理
pip install pandas polars dask

# 可视化
pip install matplotlib seaborn plotly

# 机器学习
pip install scikit-learn xgboost lightgbm

# AutoML
pip install pycaret autogluon

# 实验管理
pip install mlflow wandb
```

### 场景5: 全栈AI工程师
```bash
# 深度学习
pip install torch tensorflow

# LLM
pip install transformers langchain

# CV
pip install opencv-python ultralytics

# NLP
pip install spacy jieba

# 部署
pip install fastapi gradio docker

# MLOps
pip install mlflow dvc

# 数据处理
pip install pandas numpy
```

---

## 🎯 快速命令

### 检查安装
```python
# 检查PyTorch
import torch
print(torch.__version__)
print(torch.cuda.is_available())

# 检查TensorFlow
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# 检查Transformers
from transformers import pipeline
print("Transformers安装成功")
```

### 模型下载
```bash
# Hugging Face模型
huggingface-cli download model_name

# Ollama模型
ollama pull llama2
ollama pull mistral
```

### 环境导出
```bash
# pip
pip freeze > requirements.txt

# conda
conda env export > environment.yml
```

---

## 🔗 官方文档链接

### 框架文档
- PyTorch: https://pytorch.org/docs/
- TensorFlow: https://www.tensorflow.org/
- LangChain: https://python.langchain.com/

### 模型库
- Hugging Face: https://huggingface.co/models
- Ollama: https://ollama.ai/library

### 工具文档
- Gradio: https://www.gradio.app/docs
- FastAPI: https://fastapi.tiangolo.com/
- MLflow: https://mlflow.org/docs/

---

## 💡 使用建议

### 新手推荐
1. 先装 **PyTorch + Transformers**
2. 学习用 **Gradio** 做Demo
3. 数据处理用 **Pandas**
4. 可视化用 **Matplotlib**

### 进阶推荐
1. 部署用 **FastAPI + Docker**
2. 实验管理用 **MLflow**
3. 版本控制用 **DVC**
4. 高性能推理用 **ONNX**

### 专业推荐
1. 分布式训练：**Ray, DeepSpeed**
2. 模型优化：**TensorRT, OpenVINO**
3. 监控告警：**Prometheus, Grafana**
4. 工作流编排：**Airflow, Kubeflow**

---

## 🌟 GitHub明星项目

### 10万+ Stars
- Transformers (120k)
- Stable Diffusion WebUI (130k)
- TensorFlow (180k)

### 5万+ Stars
- PyTorch (75k)
- Ollama (60k)
- Whisper (60k)

### 2万+ Stars
- LangChain (80k)
- Ultralytics (20k+)
- Diffusers (22k+)

---

## 📚 完整列表

查看详细信息: [附录F: 开源AI工具链大全](./附录/F-开源AI工具链.md)

包含:
- ✅ 100+ 开源工具
- ✅ 12个分类
- ✅ 详细使用示例
- ✅ GitHub链接
- ✅ 安装命令

---

## 🏛️ 分层架构视图

查看完整技术栈: [附录G: AI技术栈分层架构](./附录/G-AI技术栈分层架构.md) ⭐推荐

从底层硬件到上层应用，7层完整体系：
```
第7层：应用层 (LLM/CV/NLP应用)
第6层：开发工具层 (IDE/可视化)
第5层：MLOps层 (实验管理/版本控制)
第4层：模型层 (预训练模型/AutoML)
第3层：框架层 (深度学习框架)
第2层：运行时层 (推理引擎/容器化)
第1层：基础设施层 (GPU/CUDA/云平台)
```

---

**最后更新**: 2025-10-18  
**收录工具**: 100+  
[返回目录](./README.md)
