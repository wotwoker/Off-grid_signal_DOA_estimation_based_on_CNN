import pickle
import numpy as np
import matplotlib.pyplot as plt

# 加载保存的训练历史数据
with open('history_cnn_offgrid.pkl', 'rb') as file_pi:
#with open('history_cnn_C.pkl', 'rb') as file_pi:
#with open('history_cnn_R.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)

# 设置绘图
plt.figure(figsize=[10, 4])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
# 绘制训练和验证损失
plt.subplot(1, 2, 1)
epochs = range(1, len(history['loss']) + 1)
plt.plot(epochs, history['loss'], label='网络训练损失值', linewidth=1.5, marker='s', markersize=3, markevery=4)
plt.plot(epochs, history['val_loss'], label='网络验证损失值', linewidth=1.5, marker='o', markersize=3, markevery=4)
plt.title('Training and Validation Loss', fontsize=12)
plt.xlabel('训练次数', fontsize=10)
plt.ylabel('损失值', fontsize=10)
plt.legend()
plt.grid(True)

# 绘制训练和验证的准确率 acc  acc_offgrid    micro_f1
plt.subplot(1, 2, 2)
acc_offgrid_percent= [i * 100 for i in history['acc_offgrid']]
val_acc_offgrid_percent = [i * 100 for i in history['val_acc_offgrid']]
plt.plot(epochs, acc_offgrid_percent, label='网络训练准确率', linewidth=1.5, marker='s', markersize=3, markevery=4)
plt.plot(epochs, val_acc_offgrid_percent, label='网络验证准确率', linewidth=1.5, marker='o', markersize=3, markevery=4)

# for i in range(0, len(acc_offgrid_percent), 2):
#     plt.scatter(i, acc_offgrid_percent[i], s=10, marker='s')  # 网络训练准确率的方块点
#     plt.scatter(i, val_acc_offgrid_percent[i], s=10, marker='o')  # 网络测试准确率的圆点
plt.title('Training and Validation accuracy', fontsize=12)
plt.xlabel('训练次数', fontsize=10)
plt.ylabel('准确率(%)', fontsize=10)
plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
