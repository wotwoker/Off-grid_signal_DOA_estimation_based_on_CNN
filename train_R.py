"""
输入特征为协方差矩阵 R 的训练网络
保存训练数据到 history_cnn_R.pkl文件
保存训练模型为 cnn_R.h5
"""

import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import pickle
from micro_f1 import micro_f1  # 从 micro_f1.py 中导入 micro_f1 函数
from bce_function import bce

# 加载数据
training_set = scipy.io.loadmat('train_set.mat')
Signal_eta = training_set['Signal_eta']  # shape: (8, 8, 2*nsample)
Signal_label = training_set['Signal_label']  # shape: (nsample, 181)

# 重塑Signal_eta以匹配期望的输入格式
nsample = Signal_eta.shape[2] // 2
kelm = Signal_eta.shape[0]
Signal_eta = np.transpose(Signal_eta, (2, 0, 1))  # 将Signal_eta从(8, 8, 2*nsample)转换成(2*nsample, 8, 8)
Signal_eta = Signal_eta.reshape((nsample, 2, kelm, kelm)).transpose(0, 2, 3, 1)  # 形状变化(2*nsample)->(nsample,2)+转换为(nsample, 8, 8, 2) 每个样本都是8x8的矩阵，有两个通道

# 构建模型
model_R = keras.Sequential([
    keras.layers.Conv2D(4, (3, 3), activation='relu', name="cnn_1", padding='same', input_shape=(8, 8, 2)),  # 维度2表示每个样本有两个通道
    keras.layers.Conv2D(8, (3, 3), activation='relu', name="cnn_2", padding='same'),  # 卷积核数量8 大小3x3
    keras.layers.Conv2D(16, (3, 3), activation='relu', name="cnn_3", padding='same'),
    keras.layers.Conv2D(32, (3, 3), activation='relu', name="cnn_4", padding='same'),
    keras.layers.Flatten(),  # 将卷积层的输出展平
    keras.layers.Dense(1000, activation='relu', name="dense_5"),  # 第一个全连接层
    keras.layers.Dropout(0.4),  # Dropout层，丢失率为0.3
    keras.layers.Dense(500, activation='relu', name="dense_6"),  # 第二个全连接层
    keras.layers.Dense(181, activation='sigmoid', name="dense_7")  # 输出181个二分类结果
])

# 编译模型
model_R.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                #loss='binary_crossentropy',  # 使用二元交叉熵损失函数
                loss=bce,
                metrics=[micro_f1])  # 使用Micro-F1准确率作为性能度量
                #metrics=['acc'])  # 使用准确率作为性能度量
model_R.summary()

# 训练模型
history_R = model_R.fit(Signal_eta, Signal_label, epochs=80, batch_size=256, validation_split=0.2, verbose=1)

# 保存训练历史
with open('history_cnn_R.pkl', 'wb') as file:
    pickle.dump(history_R.history, file)

# 保存模型
model_R.save('cnn_R.h5')
