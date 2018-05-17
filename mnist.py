import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import datetime
t1 = datetime.datetime.now()
with tf.device("/gpu:0"):
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	y_ = tf.placeholder("float", [None,10])
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	init = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)
	for i in range(10000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
t2 = datetime.datetime.now()
with tf.device("/cpu:0"):
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	y_ = tf.placeholder("float", [None,10])
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	init = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)
	for i in range(10000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
t3 = datetime.datetime.now()
print('gpu:'+str(t2-t1))
print('cpu:'+str(t3-t2))