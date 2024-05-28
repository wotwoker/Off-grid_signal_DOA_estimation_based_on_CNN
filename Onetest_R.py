import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
import micro_f1  # 自定义函数（ micro_f1 函数）

# 加载测试数据
# test_set = scipy.io.loadmat('OneTestSet.mat')
# theta_test = test_set['thetaOneTest'][0]  # thetaOneTest存储为一维数组
# Signal_eta = test_set['Signal_eta']  # (8, 8, 2)
# Signal_label = test_set['Signal_label'][0]  # 假设为1 x 181数组，展平
# theta = test_set['Phi'][0]  # 假设为1 x 181数组，展平

# 加载离网格信号测试数据，预测整数部分
test_set = scipy.io.loadmat('OneTestSet_offgrid.mat')
theta_test = test_set['thetaOneTest_on'][0]  # thetaOneTest存储为一维数组
Signal_eta = test_set['Signal_eta_on']  # (8, 8, 2)
Signal_label = test_set['Signal_label_on'][0]  # 假设为1 x 181数组，展平


# 为模型正确格式化Signal_eta
# 假设Signal_eta没有重新整形，因为它将直接用于预测
Signal_eta = np.expand_dims(Signal_eta, axis=0)  # 添加批次维度，现在是 (1, 8, 8, 2)

# 加载训练好的模型 # 自定义函数（ micro_f1 函数）在模型加载时需要特别指定
cnn_doa_model = tf.keras.models.load_model('cnn_R.h5', custom_objects={'micro_f1': micro_f1})

# 使用模型预测DOA
DOA_spectrum = cnn_doa_model.predict(Signal_eta)[0]  # 预测并取第一个（唯一）结果
print(DOA_spectrum)
DOA_spectrum = DOA_spectrum / np.max(DOA_spectrum)  # 归一化预测结果
print(DOA_spectrum)

"""
# 绘制DOA频谱
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 替换为你选择的字体
plt.figure(figsize=[10, 5])
plt.plot(theta, DOA_spectrum, label='CNN DOA Spectrum', linewidth=2)
plt.scatter(theta_test, np.ones_like(theta_test), color='red', label='real direction')  # 标记真实角度
plt.title('CNN-DOA空间谱')
plt.xlabel('角度（度）')
plt.ylabel('归一化谱')
plt.legend()
plt.grid(True)
plt.show()
"""
# 将预测结果保存到.mat文件中
scipy.io.savemat('cnn_predict_OneTestR.mat', {'P_cnn_R': DOA_spectrum})
