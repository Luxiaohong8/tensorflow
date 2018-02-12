import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#定义添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数

"""
所以，这里的表示方式是： input * weights 
假如，输入层的结点个数是2，隐层是3
input=[n*2]  ,weihts=[2*3] ,bias=[1,3]
input*weigths=[n,3] + bias=[1,3] ，这样的矩阵维度相加的时候，python会执行它的广播机制
so,这一层的输出的维度是 [n,3]
"""


'''
在具体的例子中, 我们默认首选的激励函数是哪些. 在少量层结构中, 我们可以尝试很多种不同的
激励函数. 在卷积神经网络 Convolutional neural networks 的卷积层中, 推荐的激励函数是 relu.
 在循环神经网络中 recurrent neural networks, 推荐的是 tanh 或者是 relu
'''

def add_layer(inputs,input_size,output_size,activation_function=None):
	with tf.name_scope('layer'):
		with tf.name_scope('weithts'):
			Weigths=tf.Variable(tf.random_normal([input_size,output_size]))
		#在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
		with tf.name_scope('biases'):
			biases=tf.Variable(tf.zeros([1,output_size])+0.1)
		#未激活的值
		with tf.name_scope('y'):
			y=tf.matmul(inputs,Weigths)+biases

		if activation_function is None:
			outputs=y
		else:
			outputs=activation_function(y)
		return outputs

#数据集
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
'''
传值的工作交给了 sess.run() , 需要传入的值放在了feed_dict={} 
并一一对应每一个 input. placeholder 与 feed_dict={} 是绑定在一起出现的。
'''


#这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1。
with tf.name_scope('input'):
	x=tf.placeholder(tf.float32,[None,1],name='x_input')
	y=tf.placeholder(tf.float32,[None,1],name='y_input')

#我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。
layer1=add_layer(x,1,20, activation_function=tf.nn.relu)

prediction=add_layer(layer1,20,1,activation_function=None)
with tf.name_scope('loss'):
	loss=tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction),reduction_indices=[1]))
with tf.name_scope('train'):
	train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter("logs/", sess.graph)
	for i in range(1000):
		sess.run(train,feed_dict={x:x_data,y:y_data})
		if i%50==0:
			print("loss:",sess.run(loss,feed_dict={x:x_data,y:y_data}))

