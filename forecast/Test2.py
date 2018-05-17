import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import datetime as dt

x_input = tf.placeholder(dtype=tf.float32, shape=[3, 3])
y_output = tf.placeholder(dtype=tf.float32, shape=[3, 1])

# 定义变量
x_weights = tf.Variable(tf.ones([3, 1]))
biases = tf.Variable(tf.ones([1, 1]))

# 构造训练集
x_data = [[2, 2, 2], [2, 2, 3], [3, 2, 4]]
y_data = np.matmul(x_data, [[1], [1], [1]])
print(y_data)

# 定义我们的预测函数
y_hat = tf.matmul(x_input, x_weights) + biases

# 定义loss函数
loss = tf.reduce_mean(tf.square(y_hat - y_output))

# 定义优化函数
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 定义Graph
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for index in range(1000):
        sess.run([train_op], feed_dict={x_input: x_data, y_output: y_data})
        print("weights:{}, biases:{}".format(sess.run(x_weights), sess.run(biases)))
