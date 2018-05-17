import tensorflow as tf
import forecast.preprocess as pp
import matplotlib.pyplot as plt
import numpy as np
from forecast.functions.data_functions import smooth_data_avg

# 1.读取销量、天气数据
data = pp.load_and_sort_data()

# 2.数据预处理
preprocessed_data = pp.preprocess_data(data)

# 3.显示处理前的数据
# plt.plot(data.dataset_train_sales_date, data.dataset_train_sales, 'r')
# plt.show()

# 4.tensorflow 线性回归天气影响
size = preprocessed_data.train_size

# 仅天气
dimension = (preprocessed_data.config.EXTEND_WEATHER_BACKWARD_DAYS
             + preprocessed_data.config.EXTEND_WEATHER_FORWARD_DAYS + 1) * 2
origin_data = preprocessed_data.dataset_train_input_weather

# 天气+销量
# dimension = (preprocessed_data.config.EXTEND_WEATHER_BACKWARD_DAYS
#              + preprocessed_data.config.EXTEND_WEATHER_FORWARD_DAYS + 1) * 2
# dimension += preprocessed_data.config.EXTEND_SALES_BACKWARD_DAYS + 1
#
# origin_data = np.concatenate(
#     (preprocessed_data.dataset_train_input_weather, preprocessed_data.dataset_train_input_sales), axis=1)

x_data = origin_data  # m x d
y_data = preprocessed_data.dataset_train_output_sales  # m x 1

print('dimension', dimension)
print('size', size)
print('x_data', x_data)
print('y_data', y_data)

# 定义W和b，这里的W是一个向量，取值范围为-1到1，b为一个数
x_input = tf.placeholder(dtype=tf.float32, shape=[None, dimension])
y_output = tf.placeholder(dtype=tf.float32, shape=[None, 1])

x_weights = tf.Variable(tf.zeros([dimension, 1]))
biases = tf.Variable(tf.zeros([1]))
# 模型
# test = anti_logistic(logistic(X * b,regress_args),sales_args) #拟合后模型，带入天气信息计算销量

# print(data.dataset_train_weather)
y_hat = tf.matmul(x_input, x_weights) + biases

# 损失函数
loss = tf.reduce_mean(tf.square(y_hat - y_output))

# 优化，基于梯度下降法的优化
train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)  # 由于特征过多，步长稍微大些就会跳过解范围，而去向无穷大

# 初始化
init_op = tf.global_variables_initializer()

# 启动图计算模型参数
with tf.Session() as sess:
    sess.run(init_op)

    times = 1000
    for index in range(times):
        if index % (times / 100) == 0:
            print('dealing:%r %%' % (index / times * 100))
        # print("BEFORE：weights:{}, biases:{}".format(sess.run(x_weights), sess.run(biases)))
        # print('BEFORE：y_hat', np.matmul(x_data, sess.run(x_weights)) + sess.run(biases))
        sess.run(train_op, feed_dict={x_input: x_data, y_output: y_data})
        # print("AFTER：weights:{}, biases:{}".format(sess.run(x_weights), sess.run(biases)))
        # 对比结果
        # print('AFTER：y_hat', np.matmul(x_data, sess.run(x_weights)) + sess.run(biases))
    w = sess.run(x_weights)
    b = sess.run(biases)
    fitting_sales = np.matmul(x_data, w) + b
    print('AFTER：y_hat', fitting_sales)

# 根据模型+预测时间段数据
actual_sales = preprocessed_data.dataset_actual_sales  # 实际销量

# # (基于当前日期之前的真实销量) -> 预测销量
# forecast_data = np.concatenate(
#     (preprocessed_data.dataset_forecast_weather, preprocessed_data.dataset_forecast_by_actual_sales), axis=1)
# preprocessed_data.dataset_forecast_sales = forecast_sales = np.matmul(forecast_data, w) + b

# 基于动态预测的销量
# 仅天气
preprocessed_data.dataset_forecast_sales = forecast_sales = \
    np.matmul(preprocessed_data.dataset_forecast_weather, w) + b

# 带预测的日销量
# forecast_sales = []
# for ii in range(len(actual_sales)):
#     # 拼接模型输入数据
#     # print("day:%r" % ii)
#     forecast_data = np.concatenate((preprocessed_data.dataset_forecast_weather[ii],
#                                     preprocessed_data.dataset_forecast_by_forecast_sales), axis=0)
#     forecast_data = smooth_data_avg(forecast_data, data.config.SMOOTH_SALES_DAYS)
#     current_day_sales = np.matmul(forecast_data, w) + b
#     preprocessed_data.dataset_forecast_by_forecast_sales = np.concatenate(
#         (preprocessed_data.dataset_forecast_by_forecast_sales[1:], current_day_sales[0]), axis=0)
#
#     forecast_sales.append(current_day_sales)

plt.plot(range(len(actual_sales)), actual_sales, 'r')
plt.plot(range(len(forecast_sales)), forecast_sales, 'g')
plt.show()

# 预测误差
actual_total_price = np.sum(actual_sales)
forecast_total_price = np.sum(forecast_sales)
print('预测总销量：')
print(forecast_total_price)
print('实际总销量：')
print(actual_total_price)
print('预测总销量/实际总销量：')
print(forecast_total_price / actual_total_price)
print('误差比 = 预测差值/实际总销量：')
print(np.abs(actual_total_price - forecast_total_price) / actual_total_price)
