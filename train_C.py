"""
输入特征为 MUSIC-doa 的训练网络
保存训练数据到 history_cnn_M.pkl 文件
保存训练模型为 cnn_M.h5
"""

import keras
import numpy as np
import scipy.io
import pickle
import tensorflow as tf

training_set = scipy.io.loadmat('train_set.mat')        # 导入训练数据
Signal_eta = training_set['Signal_eta_forC']                 # 输入特征 ____ x 181
Signal_label = training_set['Signal_label']             # 输出标签 ____ x 181
Signal_eta = np.expand_dims(Signal_eta, axis=2)          # ____ x 181 x 1
Signal_label = np.expand_dims(Signal_label, axis=2)      # ____ x 181 x 1
[Sample, P, dim] = np.shape(Signal_eta)                 # P=181个角度，dim=1

model_C = keras.Sequential()
# input_shape = (batch_size,steps,input_dim)
# filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
# kernel_size: 一个整数，或者单个整数表示的元组或列表，指明1D卷积窗口的长度。
# padding='same'：表示填充输入以使输出具有与原始输入相同的长度。
# output_shape = (batch_size,new_steps,filters)
# new_steps = (steps-kernel_size+2*padding)+1
# Param = filters * kernal_size * new_filters + new_filters
model_C.add(keras.layers.Convolution1D(filters=12, kernel_size=40, input_shape=(P, dim), activation='ReLU', name="cnn_1", padding='same'))
model_C.add(keras.layers.Convolution1D(filters=6, kernel_size=20, activation='ReLU', name="cnn_2", padding='same'))
model_C.add(keras.layers.Convolution1D(filters=3, kernel_size=10, activation='ReLU', name="cnn_3", padding='same'))
model_C.add(keras.layers.Convolution1D(filters=1, kernel_size=5, activation='ReLU', name="cnn_4", padding='same'))
model_C.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse', metrics = ['acc'])
model_C.summary()

history_C = model_C.fit(x=Signal_eta,  # ~x181x1，输入空间谱
                        y=Signal_label,  # ~x181x1，输出标签，存在目标位置为1，否则为0
                        batch_size=256,  # 1次网络权重更新所使用的样本量
                        epochs=20,  # 训练的总轮数
                        verbose=1, validation_split=0.2, shuffle=True  # 随机抽样
                        )
# 保存训练历史
with open('history_cnn_C.pkl', 'wb') as file_pi:
    pickle.dump(history_C.history, file_pi)

# 保存模型
model_C.save('cnn_C.h5')
