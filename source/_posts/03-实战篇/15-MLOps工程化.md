# 第15章：MLOps工程化

## 📝 本章目标
- 理解MLOps理念
- 掌握实验管理工具
- 学习CI/CD流程
- 实现模型监控

## 15.1 MLflow实验管理

```python
import mlflow
import mlflow.sklearn

# 记录实验
with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_metric("rmse", 0.789)
    mlflow.sklearn.log_model(model, "model")
```

## 15.2 DVC数据版本控制

```bash
# 初始化DVC
dvc init

# 跟踪数据
dvc add data/dataset.csv
git add data/dataset.csv.dvc .gitignore
git commit -m "Add dataset"

# 推送到远程存储
dvc remote add -d storage s3://mybucket/dvcstore
dvc push
```

## 15.3 模型监控

```python
from prometheus_client import Counter, Histogram
import time

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@prediction_duration.time()
def predict(data):
    prediction_counter.inc()
    result = model.predict(data)
    return result
```

## 15.4 A/B测试

```python
class ABTester:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
    
    def predict(self, user_id, data):
        if hash(user_id) % 100 < self.split_ratio * 100:
            return self.model_a.predict(data), 'A'
        else:
            return self.model_b.predict(data), 'B'
```

## 📚 本章小结
- ✅ MLflow实验跟踪
- ✅ DVC数据版本控制
- ✅ 模型监控系统
- ✅ A/B测试框架

[⬅️ 上一章](./14-模型部署.md) | [返回目录](../README.md) | [下一章 ➡️](../04-商业应用篇/16-推荐系统.md)
