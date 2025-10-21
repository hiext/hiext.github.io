# 附录A：开发环境配置指南

## 🎯 本章目标

- 配置Python开发环境
- 安装深度学习框架（TensorFlow、PyTorch）
- 配置GPU加速（CUDA）
- 搭建开发工具链（IDE、Jupyter）

---

## A.1 Python环境配置

### A.1.1 安装Anaconda/Miniconda

**推荐使用Miniconda**（更轻量）

#### Windows系统

```bash
# 1. 下载Miniconda
# 访问: https://docs.conda.io/en/latest/miniconda.html

# 2. 安装后，打开Anaconda Prompt

# 3. 配置国内镜像源（加速下载）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes

# 4. 创建虚拟环境
conda create -n ai-dev python=3.10
conda activate ai-dev
```

#### Linux/macOS系统

```bash
# 1. 下载并安装
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. 配置镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --set show_channel_urls yes

# 3. 创建环境
conda create -n ai-dev python=3.10
conda activate ai-dev
```

### A.1.2 配置pip镜像源

```bash
# Linux/macOS
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# Windows (PowerShell)
$pipDir = "$env:APPDATA\pip"
if (-not (Test-Path $pipDir)) { New-Item -ItemType Directory -Path $pipDir }

@"
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
"@ | Out-File -FilePath "$pipDir\pip.ini" -Encoding ASCII
```

---

## A.2 深度学习框架安装

### A.2.1 TensorFlow安装

```bash
# CPU版本
pip install tensorflow

# GPU版本（需要先安装CUDA，见下文）
pip install tensorflow-gpu

# 验证安装
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

### A.2.2 PyTorch安装

访问官网选择配置：https://pytorch.org/get-started/locally/

```bash
# CPU版本
pip install torch torchvision torchaudio

# GPU版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU版本（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### A.2.3 常用科学计算库

```bash
# 基础库
pip install numpy pandas matplotlib seaborn scipy

# 机器学习
pip install scikit-learn xgboost lightgbm

# 深度学习辅助库
pip install opencv-python pillow albumentations

# NLP相关
pip install transformers tokenizers datasets

# 可视化
pip install plotly tensorboard wandb

# Web开发
pip install fastapi uvicorn gradio streamlit

# 数据库
pip install sqlalchemy pymongo redis

# 工具库
pip install tqdm joblib pyyaml python-dotenv
```

---

## A.3 CUDA与GPU配置

### A.3.1 检查GPU信息

```bash
# Windows
nvidia-smi

# Linux
lspci | grep -i nvidia
nvidia-smi
```

### A.3.2 安装CUDA Toolkit

**方式一：从NVIDIA官网下载**

1. 访问：https://developer.nvidia.com/cuda-downloads
2. 选择操作系统和版本
3. 下载并安装

**推荐版本**：
- CUDA 11.8（稳定性好）
- CUDA 12.1（最新）

### A.3.3 安装cuDNN

1. 访问：https://developer.nvidia.com/cudnn
2. 注册NVIDIA账号
3. 下载对应CUDA版本的cuDNN
4. 解压并复制文件到CUDA目录

```bash
# Linux示例
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### A.3.4 配置环境变量

**Linux/macOS**

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 重新加载
source ~/.bashrc
```

**Windows**

1. 右键"此电脑" → "属性" → "高级系统设置"
2. "环境变量" → "系统变量"
3. 添加到Path：
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp`

### A.3.5 验证CUDA安装

```bash
# 查看版本
nvcc --version

# 运行示例
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

---

## A.4 开发工具配置

### A.4.1 VS Code配置

**安装VS Code**

访问：https://code.visualstudio.com/

**推荐插件**：

```
必装:
- Python (Microsoft)
- Pylance
- Jupyter

推荐:
- Python Docstring Generator
- autoDocstring
- GitLens
- Markdown All in One
- Better Comments
- Error Lens
```

**配置文件（settings.json）**：

```json
{
    "python.defaultInterpreterPath": "C:\\Users\\YourName\\miniconda3\\envs\\ai-dev\\python.exe",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "jupyter.askForKernelRestart": false,
    "files.autoSave": "afterDelay"
}
```

### A.4.2 PyCharm配置

**安装PyCharm**

- Professional版（学生可免费）
- Community版（开源免费）

**配置Python解释器**：

1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Conda Environment
3. 选择ai-dev环境

### A.4.3 Jupyter配置

```bash
# 安装Jupyter
pip install jupyter jupyterlab notebook

# 安装扩展
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# 启动Jupyter Lab
jupyter lab

# 启动Jupyter Notebook
jupyter notebook

# 配置密码
jupyter notebook password
```

**Jupyter主题美化**：

```bash
pip install jupyterthemes

# 应用主题
jt -t onedork -fs 12 -altp -tfs 11 -nfs 115 -cellw 88% -T
```

---

## A.5 Docker环境（可选）

### A.5.1 安装Docker

**Windows/Mac**：
- 下载Docker Desktop：https://www.docker.com/products/docker-desktop

**Linux**：

```bash
# Ubuntu
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### A.5.2 使用官方深度学习镜像

```bash
# TensorFlow GPU
docker pull tensorflow/tensorflow:latest-gpu-jupyter

# PyTorch
docker pull pytorch/pytorch:latest

# 运行容器
docker run -it --gpus all -p 8888:8888 \
  -v $(pwd):/workspace \
  tensorflow/tensorflow:latest-gpu-jupyter
```

### A.5.3 自定义Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安装Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装深度学习框架
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    tensorflow \
    transformers \
    jupyter

WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

---

## A.6 常见问题解决

### Q1: CUDA版本不匹配

```bash
# 查看CUDA版本
nvidia-smi  # 查看驱动支持的最高CUDA版本
nvcc --version  # 查看安装的CUDA版本

# 解决方案：安装兼容的PyTorch版本
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### Q2: 内存不足

```python
# PyTorch - 限制GPU内存使用
import torch
torch.cuda.set_per_process_memory_fraction(0.5, 0)  # 使用50%

# TensorFlow - 动态内存分配
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Q3: pip安装速度慢

```bash
# 临时使用国内源
pip install package_name -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用代理
pip install package_name --proxy http://proxy.example.com:8080
```

### Q4: ImportError

```bash
# 重新安装包
pip uninstall package_name
pip install package_name --no-cache-dir

# 检查依赖
pip check
```

---

## A.7 完整环境脚本

### Windows环境一键安装脚本

```powershell
# setup_ai_env.ps1

# 创建conda环境
conda create -n ai-dev python=3.10 -y
conda activate ai-dev

# 安装深度学习框架
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow

# 安装常用库
pip install numpy pandas matplotlib seaborn scikit-learn
pip install transformers datasets
pip install jupyter jupyterlab
pip install fastapi uvicorn gradio

Write-Host "环境安装完成！" -ForegroundColor Green
```

### Linux环境一键安装脚本

```bash
#!/bin/bash
# setup_ai_env.sh

set -e

echo "🚀 开始配置AI开发环境..."

# 创建conda环境
conda create -n ai-dev python=3.10 -y
source activate ai-dev

# 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装TensorFlow
pip install tensorflow

# 安装常用库
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install transformers datasets tokenizers
pip install opencv-python pillow
pip install jupyter jupyterlab notebook
pip install fastapi uvicorn gradio streamlit
pip install tqdm joblib pyyaml python-dotenv

echo "✅ 环境配置完成！"
echo "使用 'conda activate ai-dev' 激活环境"
```

---

## 📚 参考资源

- **PyTorch官方文档**：https://pytorch.org/docs/
- **TensorFlow官方文档**：https://www.tensorflow.org/
- **CUDA文档**：https://docs.nvidia.com/cuda/
- **Conda文档**：https://docs.conda.io/

---

[返回目录](../README.md)
