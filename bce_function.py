
import tensorflow as tf
# 其调用是由 TensorFlow 的计算图控制的，而不是 Python 控制流


def bce_offgrid(y_ture, y_pred):  # 对y_pred正则化的BCE
    #print(p)
    # #print(y_pred)
    # p = tf.cast(p, tf.float32)
    # y_pred = tf.cast(y_pred, tf.float32)
    # epsilon = tf.keras.backend.epsilon()  # 为避免log(0)导致的数值不稳定，使用小的epsilon值进行平滑处理
    # y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    # y_logits = tf.math.log(y_pred / (1 - y_pred))
    # g_z = tf.abs((y_logits+1)/(p+1+epsilon)-1)  # 加epsilon防止除零
    z = y_pred  # z（pred） 应为模型输出值  -0.5~0.5
    p = y_ture  # p（true） 为小数标签 -1 或者 -0.5~0.5
    g_z = tf.abs((z + 0.56) / (p + 0.56) - 1)  # 防止除零
    g_z = tf.where(p != -1, g_z, tf.zeros_like(g_z))  # p!=-1时g_z值保留，否则为0 （没这一行会有loss=nan,acc=0
    sigm_gz = 1 / (1+tf.exp(g_z))
    sigm_z = 1/(1+tf.exp(-z))
    bce = -tf.cast(p != -1, tf.float32) * tf.math.log(sigm_gz) \
          - tf.cast(p == -1, tf.float32) * tf.math.log(1 - sigm_z)
    return tf.reduce_mean(bce)  # 计算输入张量所有元素的平均值，返回一个标量


def bce(y_true, y_pred ):  # 标准BCE
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()  # 为避免log(0)导致的数值不稳定，使用小的epsilon值进行平滑处理
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    return tf.reduce_mean(bce)

# 控制台测试
# import tensorflow as tf
# y = tf.constant([1, -1, 4, 4, -1])  # 用来创建一个TensorFlow张量
# signal_indices = tf.where(y != -1)
# b = 2*y
# c=b*y
# d = tf.where(y != -1, c, tf.zeros_like(c))
