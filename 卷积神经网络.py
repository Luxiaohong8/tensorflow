import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

#把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量
#因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
x_image=tf.reshape(xs,[-1,28,28,1])

#建立卷积层，卷积层的内核大小是5x5，黑白图片channel1是1输入为1，输出32
w_conv1=weight_variable([5,5,1,32]) #patch大小5x5
b_conv1=bias_variable([32])
#卷积运算
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64   ？？？ 64 是什么意思？
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

#全连接层
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	for i in range(500):
		batch_xs,batch_ys=mnist.train.next_batch(100)
		sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
		if i%50==0:
			print(compute_accuracy(mnist.test.images[:1000],mnist.test.labels[:1000]))
	res=sess.run(prediction,feed_dict={xs: mnist.test.images[:20],keep_prob:1})
	print ("number is : ", sess.run(tf.argmax(res,1)))

f, a = plt.subplots(2, 20, figsize=(10, 2))


for i in range(20):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.show()
