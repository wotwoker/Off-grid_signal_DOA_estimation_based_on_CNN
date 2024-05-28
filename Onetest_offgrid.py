import numpy as np
import scipy.io
import tensorflow as tf

from micro_f1 import acc_offgrid  # 自定义函数（ micro_f1 函数）
from bce_function import bce_offgrid

# 加载测试数据
test_set = scipy.io.loadmat('OneTestSet_offgrid.mat')
theta_test = test_set['thetaOneTest_off'][0]  # thetaOneTest存储为一维数组
Signal_eta = test_set['Signal_eta_off']  # (8, 8, 2)

# 抽取训练数据
# n = 0
# test_set = scipy.io.loadmat('trainoff_set.mat')
# theta_test = test_set['theta_train'][:, n]  # 2xsample抽第n+1个样本
# Signal_eta = test_set['Signal_eta'][:, :, n:n+2]  # (8, 8, 2)
# Signal_label = test_set['Signal_label'][n]

# 为模型正确格式化Signal_eta
# Signal_eta没有重新整形，将它直接用于预测
Signal_eta = np.expand_dims(Signal_eta, axis=0)  # 添加批次维度，现在是 (1, 8, 8, 2)

# 加载训练好的模型 # 自定义函数（ micro_f1 函数）在模型加载时需要特别指定
cnn_doa_model = tf.keras.models.load_model('cnn_offgrid.h5', custom_objects={'acc_offgrid': acc_offgrid,
                                                                             'bce_offgrid': bce_offgrid})

# 使用模型预测DOA
DOA_spectrum = cnn_doa_model.predict(Signal_eta)[0]  # 预测并取第一个（唯一）结果
DOA_spectrum_tanh = np.tanh(DOA_spectrum)
print(DOA_spectrum)
print(DOA_spectrum_tanh)
#DOA_spectrum = DOA_spectrum / np.max(DOA_spectrum)  # 这里不应该归一化


# 将预测结果保存到.mat文件中
scipy.io.savemat('cnn_predict_OneTest_offgrid.mat', {'P_cnn_offgrid': DOA_spectrum,
                                                     'P_cnn_offgrid_tanh': DOA_spectrum_tanh})
