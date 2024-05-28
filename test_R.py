# 批量估计测试集的DOA，并保存至cnn_R_predict.mat
import keras
import numpy as np
import scipy.io
import micro_f1  # 自定义函数（ micro_f1 函数）

# 加载测试数据
#test_program = scipy.io.loadmat('test_set_snr.mat') # 导入测试集
test_program = scipy.io.loadmat('test_set_snr.mat') # 导入测试集
Signal_eta = test_program['Signal_eta']  # (variable, 8, 8, 2*nsample)

# 从Signal_eta中提取必要的维度信息
P = 181
nVariable = Signal_eta.shape[0]
nsample = Signal_eta.shape[3] // 2
kelm = Signal_eta.shape[1]

# 重塑Signal_eta以匹配模型输入格式 (nsample, 8, 8, 2)
Signal_eta = np.transpose(Signal_eta, (0, 3, 1, 2))  # 将Signal_eta从(variable,8,8,2*nsample)转换成(variable,2*nsample,8,8)
Signal_eta = Signal_eta.reshape((nVariable, nsample, 2, kelm, kelm)).transpose(0, 1, 3, 4, 2)  # (variable,nsample,8,8,2)

# 初始化存放预测结果的数组
est_cnn_R = np.zeros((nVariable, nsample, P))  # DOA空间谱估计结果，(variable,nsample,P)
# 对每个变量和每个样本进行预测
cnn_doa_model = keras.models.load_model('cnn_R.h5', custom_objects={'micro_f1': micro_f1})
for iVariable in range(nVariable):
    for iSample in range(nsample):
        # 预测一个样本 (1, 8, 8, 2)
        sample_to_predict = Signal_eta[iVariable, iSample].reshape(1, kelm, kelm, 2)
        est_cnn_R[iVariable, iSample, :] = cnn_doa_model.predict(sample_to_predict)

scipy.io.savemat('cnn_predict_R.mat', {'estCNN_R': est_cnn_R})
