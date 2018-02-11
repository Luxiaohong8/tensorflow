import tensorflow as tf
import numpy as np

#create data
x_data=np.random.rand(100).astype(np.float32)
#创建大小为100的随机数据集【0，1】
y_data=x_data*0.3*x_data+0.3

#print(x_data,y_data)

# 随机初始化 权重
weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

#估计值计算方式
y=weight*x_data*x_data+biases

#计算cost
loss=tf.reduce_mean(tf.square(y-y_data))

#进行梯度下降优化
optimizer=tf.train.GradientDescentOptimizer(0.5)  # 0.5 学习率
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # 替换成这样就好

#训练
with tf.Session() as sess:
	sess.run(init)
	#print (x_data)
	#print (y_data)
	for step in range(201):
		sess.run(train)
		if step%20==0:
			print(step, sess.run(weight), sess.run(biases))
