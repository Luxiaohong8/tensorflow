"""
搭建一个最简单的网络，只有输入层和输出层
输入数据的维度是 28*28 = 784 
输出数据的维度是 10个特征
激活函数使用softmax
"""

# number 1 to 10 data
# one_hot 表示的是一个长度为n的数组，只有一个元素是1.0，其他元素都是0.0 
# 比如在n=4的情况下，标记2对用的one_hot 的标记是 [0.0 , 0.0 , 1.0 ,0.0]
# 使用 one_hot 的直接原因是，我们使用 0～9 个类别的多分类的输出层是 softmax 层
# softmax 它的输 出是一个概率分布，从而要求输入的标记也以概率分布的形式出现，进而可以计算交叉熵
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,input_size,output_size,activation_function=None):
	Weigths=tf.Variable(tf.random_normal([input_size,output_size]))
	#在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
	biases=tf.Variable(tf.zeros([1,output_size])+0.1)
	#未激活的值
	y=tf.matmul(inputs,Weigths)+biases

	if activation_function is None:
		outputs=y
	else:
		outputs=activation_function(y)
	return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # tf.argmax (y_pre,1 ) 返回每一行 下标最大的元素，1表示按行
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


xs=tf.placeholder(tf.float32,[None,784])#28x28
ys=tf.placeholder(tf.float32,[None,10])
'''
layer1=add_layer(xs,784,100, activation_function=tf.nn.softmax)
prediction=add_layer(layer1,100,10,activation_function=tf.nn.softmax)
'''

'''
layer1=add_layer(xs,784,144, activation_function=tf.nn.softmax)
layer2=add_layer(layer1,144,100, activation_function=tf.nn.softmax)
prediction=add_layer(layer2,100,10,activation_function=tf.nn.softmax)
'''
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)
# loss函数（即最优化目标函数）选用交叉熵函数
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零
corss_entroy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(corss_entroy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		biatch_xs,biatch_ys=mnist.train.next_batch(100)
		sess.run(train_step,feed_dict={xs:biatch_xs,ys:biatch_ys})
		if i%50==0:
			print(compute_accuracy(mnist.test.images,mnist.test.labels))

	res=sess.run(prediction,feed_dict={xs: mnist.test.images[:20]})
	print ("number is : ", sess.run(tf.argmax(res,1)))

f, a = plt.subplots(2, 20, figsize=(10, 2))


for i in range(20):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.show()