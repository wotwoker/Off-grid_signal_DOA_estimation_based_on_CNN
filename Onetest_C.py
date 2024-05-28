# 估计单个测试集
import keras
import numpy as np
import scipy.io

test_set = scipy.io.loadmat('OneTestSet.mat')  # 导入训练数据
theta_test = test_set['thetaOneTest']  # target x 1
Signal_eta = test_set['Signal_eta_forC']  # 1 x 181
Signal_label = test_set['Signal_label']  # 1 x 181
Signal_eta = np.expand_dims(Signal_eta, axis=2) # 1x181x1
Signal_label = np.expand_dims(Signal_label, axis=2)  # 1x181x1
[sample, P, dim] = np.shape(Signal_eta)  # 1个样本，P=181个角度，dim=1

cnn_doa_last = keras.models.load_model('cnn_C.h5')
DOA_spectrum = np.zeros((sample, P))  # DOA空间谱估计结果，1x181

DOA_spectrum = cnn_doa_last.predict(Signal_eta)
DOA_spectrum = DOA_spectrum.reshape(1, P)
DOA_spectrum = DOA_spectrum / np.max(DOA_spectrum)
print(DOA_spectrum)

scipy.io.savemat('cnn_predict_OneTestC.mat', {'P_cnn_C': DOA_spectrum})

