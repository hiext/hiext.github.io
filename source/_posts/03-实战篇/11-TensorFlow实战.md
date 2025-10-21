# 第11章：TensorFlow实战

## 📝 本章目标
- 掌握TensorFlow 2.x核心API
- 学习Keras高级功能
- 实现完整训练流程
- 掌握模型保存和加载

## 11.1 TensorFlow基础

```python
import tensorflow as tf

# 张量创建
a = tf.constant([1, 2, 3])
b = tf.Variable([4, 5, 6])
c = tf.zeros((3, 3))
d = tf.ones((2, 2))

# 张量运算
x = tf.constant([[1, 2], [3, 4]])
y = tf.constant([[5, 6], [7, 8]])

print(tf.add(x, y))
print(tf.matmul(x, y))
print(tf.reduce_mean(x))
```

## 11.2 Keras Sequential API

```python
from tensorflow import keras
from tensorflow.keras import layers

# Sequential模型
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

## 11.3 Functional API

```python
# 多输入多输出模型
input1 = keras.Input(shape=(784,))
input2 = keras.Input(shape=(10,))

x1 = layers.Dense(64, activation='relu')(input1)
x2 = layers.Dense(32, activation='relu')(input2)

merged = layers.concatenate([x1, x2])
output = layers.Dense(10, activation='softmax')(merged)

model = keras.Model(inputs=[input1, input2], outputs=output)
```

## 11.4 自定义层和模型

```python
class CustomLayer(layers.Layer):
    def __init__(self, units):
        super(CustomLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# 自定义模型
class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
```

## 11.5 训练循环

```python
# 自定义训练循环
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_acc_metric(y, predictions)
    return loss

# 训练循环
for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
    
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
```

## 11.6 模型保存和加载

```python
# 保存整个模型
model.save('my_model.h5')
model = keras.models.load_model('my_model.h5')

# 只保存权重
model.save_weights('my_weights.h5')
model.load_weights('my_weights.h5')

# SavedModel格式
model.save('saved_model/')
model = keras.models.load_model('saved_model/')
```

## 11.7 TensorBoard可视化

```python
# TensorBoard回调
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1
)

model.fit(
    x_train, y_train,
    epochs=10,
    callbacks=[tensorboard_callback]
)

# 启动TensorBoard
# tensorboard --logdir=./logs
```

## 📚 本章小结
- ✅ TensorFlow基础操作
- ✅ Keras API(Sequential, Functional)
- ✅ 自定义层和模型
- ✅ 训练循环实现
- ✅ 模型保存和可视化

[⬅️ 上一章](../02-进阶篇/10-自然语言处理.md) | [返回目录](../README.md) | [下一章 ➡️](./12-PyTorch实战.md)
