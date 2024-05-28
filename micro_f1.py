import tensorflow as tf
from keras import backend as K  # 即使IDE显示红色错误，代码本身是没有问题的md


# 定义micro-F1分数作为性能度量
def micro_f1(y_true, y_pred):
    """计算微平均F1分数"""
    y_pred = K.cast(K.greater(y_pred, 0.3), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)  # tf.where 函数检查并替换任何的 NaN 值为 0
    return K.mean(f1)

def acc_offgrid(y_true, y_pred):
    # 对于有信号的角度，计算预测值与真实值之间的差异
    threshold = 0.05  # 设置一个阈值来判断预测的DOA小数部分是否足够接近真实的小数部分
    signal_indices = tf.where(y_true != -1)  # 找到有信号的角度位置
    signal_diff = tf.gather_nd(tf.abs(y_true - y_pred), signal_indices)
    correct_signal = tf.less_equal(signal_diff, threshold)  # 检查差异是否在阈值之内
    signal_accuracy = tf.reduce_mean(tf.cast(correct_signal, tf.float32))  # 计算有信号的角度的准确率
    # 对于无信号的角度，确保模型没有错误地预测出信号
    no_signal_indices = tf.where(y_true == -1)
    no_signal_predictions = tf.gather_nd(y_pred, no_signal_indices)
    no_signal_correct = tf.less(no_signal_predictions, -15)   # 假设无信号的阈值判断为预测值小于0.5（可能需要根据实际调整）
    no_signal_accuracy = tf.reduce_mean(tf.cast(no_signal_correct, tf.float32))   # 计算无信号的角度的准确率
    # 计算有信号和无信号的角度的数量
    num_signal = tf.cast(tf.size(signal_indices), tf.float32)  # 从int32转成浮点数float32
    num_no_signal = tf.cast(tf.size(no_signal_indices), tf.float32)
    # 综合有信号和无信号的准确率
    total = num_signal + num_no_signal
    overall_accuracy = (num_signal*signal_accuracy + num_no_signal*no_signal_accuracy) / total
    #overall_accuracy = (signal_accuracy + no_signal_accuracy) / 2
    return overall_accuracy