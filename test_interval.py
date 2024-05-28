# 估计等间隔测试集的DOA，
# 保存至cnn_predict_ITVC.mat和cnn_predict_ITVR.mat和cnn_predict_ITVoff.mat

import keras
import numpy as np
import scipy.io
import micro_f1  # 自定义函数（ micro_f1 函数）
from micro_f1 import acc_offgrid  # 自定义函数（ micro_f1 函数）
from bce_function import bce_offgrid

test_program = scipy.io.loadmat('test_set_interval.mat') #导入测试集
Signal_eta_C = test_program['Signal_eta_forC'] #300x180，输入空间谱
Signal_eta_R = test_program['Signal_eta_on']  # (8, 8, 2*nsample)
Signal_eta_off = test_program['Signal_eta']  # (8, 8, 2)



[sample, P] = np.shape(Signal_eta_C)  # 300x181
est_cnn_C = np.zeros((sample, P))  # DOA空间谱估计结果，300x181
cnn_doa_C = keras.models.load_model('cnn_C.h5')
for iSample in range(sample):
    est = cnn_doa_C.predict(Signal_eta_C[iSample, :].reshape(1, P, 1))
    est_cnn_C[iSample, :] = est.reshape(1, P)
scipy.io.savemat('cnn_predict_ITVC.mat', {'estCNN_C': est_cnn_C})



# 从Signal_eta中提取必要的维度信息
kelm = Signal_eta_R.shape[1]
# 重塑Signal_eta以匹配模型输入格式 (sample, 8, 8, 2)
Signal_eta_R = np.transpose(Signal_eta_R, (2, 0, 1))  # 将Signal_eta从(8, 8, 2*sample)转换成(2*sample, 8, 8)
Signal_eta_R = Signal_eta_R.reshape((sample, 2, kelm, kelm)).transpose(0, 2, 3, 1)  # 形状变化(2*sample)->(sample,2)+转换为(sample, 8, 8, 2) 每个样本都是8x8的矩阵，有两个通道
# 初始化存放预测结果的数组
est_cnn_R = np.zeros((sample, P))  # DOA空间谱估计结果，(sample,P)
# 对每个变量和每个样本进行预测
cnn_doa_R = keras.models.load_model('cnn_R.h5', custom_objects={'micro_f1': micro_f1})
for i in range(sample):
    # 预测一个样本 (1, 8, 8, 2)
    sample_to_predict = Signal_eta_R[i].reshape(1, kelm, kelm, 2)
    est_cnn_R[i, :] = cnn_doa_R.predict(sample_to_predict)
scipy.io.savemat('cnn_predict_ITVR.mat', {'estCNN_R': est_cnn_R})



# 为模型正确格式化Signal_eta
# Signal_eta没有重新整形，将它直接用于预测
Signal_eta_off = np.transpose(Signal_eta_off, (2, 0, 1))
Signal_eta_off = Signal_eta_off.reshape((sample, 2, kelm, kelm)).transpose(0, 2, 3, 1)
est_cnn_off = np.zeros((sample, P))
cnn_doa_off = keras.models.load_model('cnn_offgrid.h5', custom_objects={'acc_offgrid': acc_offgrid,
                                                                             'bce_offgrid': bce_offgrid})
for i in range(sample):
    sample_to_predict = Signal_eta_off[i].reshape(1, kelm, kelm, 2)
    DOA_spectrum = cnn_doa_off.predict(sample_to_predict)  # 预测并取第一个（唯一）结果
    est_cnn_off[i, :] = np.tanh(DOA_spectrum)
scipy.io.savemat('cnn_predict_ITVoff.mat', {'estCNN_off': est_cnn_off})