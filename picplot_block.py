import numpy as np
import matplotlib.pyplot as plt

# # 定义sigmoid函数
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# # 定义sigmoid函数的导数
# def sigmoid_derivative(x):
#     s = sigmoid(x)
#     return s * (1 - s)
# # 创建一个x值的范围以绘制函数
# x = np.linspace(-10, 10, 200)
# # 设置图形大小
# plt.figure(figsize=(10, 4))
# # 绘制sigmoid函数
# plt.subplot(1, 2, 1)
# plt.plot(x, sigmoid(x), 'b', label='sigmoid', linewidth=2)
# #plt.title('Sigmoid Function')
# plt.xlabel('x')
# plt.ylabel('sigmoid(x)')
# plt.axhline(0, color='gray', linewidth=0.5)
# plt.axvline(0, color='gray', linewidth=0.5)
# plt.grid(True)
# plt.legend()
# # 绘制sigmoid函数的导数
# plt.subplot(1, 2, 2)
# plt.plot(x, sigmoid_derivative(x), 'r', label="sigmoid'", linewidth=2)
# #plt.title('Derivative of Sigmoid Function')
# plt.xlabel('x')
# plt.ylabel('sigmoid\'(x)')
# plt.axhline(0, color='gray', linewidth=0.5)
# plt.axvline(0, color='gray', linewidth=0.5)
# plt.grid(True)
# plt.legend()
# # 调整子图间的间距
# plt.subplots_adjust(wspace=0.4)
# # 显示图像
# plt.show()


# # 定义ReLU函数
# def relu(x):
#     return np.maximum(0, x)
# # 定义ReLU函数的导数
# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)
# # 创建一个x值的范围以绘制函数
# x = np.linspace(-10, 10, 200)
# # 设图形大小
# plt.figure(figsize=(10, 4))
# # 绘制ReLU函数
# plt.subplot(1, 2, 1)
# plt.plot(x, relu(x), 'b', label='ReLU', linewidth=2)
# #plt.title('ReLU Function')
# plt.xlabel('x')
# plt.ylabel('ReLU(x)')
# plt.axhline(0, color='gray', linewidth=0.5)
# plt.axvline(0, color='gray', linewidth=0.5)
# plt.grid(True)
# plt.legend()
# # 绘制ReLU函数的导数
# plt.subplot(1, 2, 2)
# plt.plot(x, relu_derivative(x), 'r', label='ReLU\'', linewidth=2)
# #plt.title('Derivative of ReLU Function')
# plt.xlabel('x')
# plt.ylabel('ReLU\'(x)')
# plt.axhline(0, color='gray', linewidth=0.5)
# plt.axvline(0, color='gray', linewidth=0.5)
# plt.grid(True)
# plt.legend()
# # 调整子图间的间距
# plt.subplots_adjust(wspace=0.4)
# # 显示图像
# plt.show()



# # 定义tanh函数
# def tanh(x):
#     return np.tanh(x)
# # 定义tanh函数的导数
# def tanh_derivative(x):
#     return 1 - np.tanh(x)**2
# # 创建一个x值的范围以绘制函数
# x = np.linspace(-10, 10, 200)
# # 设置图形大小
# plt.figure(figsize=(10, 4))
# # 绘制tanh函数
# plt.subplot(1, 2, 1)
# plt.plot(x, tanh(x), 'b', label='tanh', linewidth=2)
# plt.title('tanh Function')
# plt.xlabel('x')
# plt.ylabel('tanh(x)')
# plt.axhline(0, color='gray', linewidth=0.5)
# plt.axvline(0, color='gray', linewidth=0.5)
# plt.grid(True)
# plt.legend()
# # 绘制tanh函数的导数
# plt.subplot(1, 2, 2)
# plt.plot(x, tanh_derivative(x), 'r', label="tanh'", linewidth=2)
# plt.title('Derivative of tanh Function')
# plt.xlabel('x')
# plt.ylabel('tanh\'(x)')
# plt.axhline(0, color='gray', linewidth=0.5)
# plt.axvline(0, color='gray', linewidth=0.5)
# plt.grid(True)
# plt.legend()
# # 调整子间的间距
# plt.subplots_adjust(wspace=0.4)
# # 显示图像
# plt.show()
