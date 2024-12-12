# tensorflow2 实现mnist数据集的模型，并保存模型,导出为SavedModel格式，并进行 预测等

import tensorflow as tf
from tensorflow.keras import layers, models

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train, x_test = x_train / 255.0, x_test / 255.0

# 定义简单的卷积神经网络模型
model = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),  # MNIST 图片是28x28，单通道
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10类数字
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")


# 将模型保存为 SavedModel 格式
saved_model_path = "./mnist_model"
model.save(saved_model_path, save_format='tf')
print(f"Model saved at {saved_model_path}")


# 加载保存的模型
loaded_model = tf.keras.models.load_model(saved_model_path)

# 打印加载模型的结构
loaded_model.summary()

# 测试加载的模型
test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"Loaded model test accuracy: {test_acc}")

import numpy as np

# 随机从测试集中取一张图片
random_idx = np.random.randint(0, len(x_test))
random_image = x_test[random_idx].reshape(1, 28, 28, 1)  # 需要将图片reshape成模型输入的形式

# 使用模型进行预测
predictions = loaded_model.predict(random_image)
predicted_label = np.argmax(predictions)

print(f"Predicted label: {predicted_label}")
print(f"True label: {y_test[random_idx]}")
