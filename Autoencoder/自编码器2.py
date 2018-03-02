import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate=0.01
training_epochs=5
batch_size=256
display_step=1
examples_show=10

n_input=784
X= tf.placeholder("float",[None,n_input])
'''
在压缩环节：我们要把这个Features不断压缩，经过第一个隐藏层压缩至256个 
	Features，再经过第二个隐藏层压缩至128个。
在解压环节：我们将128个Features还原至256个，再经过一步还原至784个。
在对比环节：比较原始数据与还原后的拥有 784 Features 的数据进行 cost 
	的对比，根据 cost 来提升我的 Autoencoder 的准确率，下图是两个隐藏层的
	weights 和 biases 的定义：
'''

n_hidden_1=256
n_hidden_2=128
n_hidden_3=64
n_hidden_4=8

weights = {
	'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
	'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4,n_hidden_3])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3,n_hidden_2])),
	'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
	'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
	}
biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
	'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b4': tf.Variable(tf.random_normal([n_input])),
	}

'''
计算 x 元素的sigmoid。

具体来说，就是：y = 1/(1 + exp (-x))
'''

def encoder(x):
	layer1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
	layer2=tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['encoder_h2']),biases['encoder_b2']))
	layer3=tf.nn.sigmoid(tf.add(tf.matmul(layer2,weights['encoder_h3']),biases['encoder_b3']))
	layer4=tf.nn.sigmoid(tf.add(tf.matmul(layer3,weights['encoder_h4']),biases['encoder_b4']))
	return layer4

def decoder(x):
	layer1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
	layer2=tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['decoder_h2']),biases['decoder_b2']))
	layer3=tf.nn.sigmoid(tf.add(tf.matmul(layer2,weights['decoder_h3']),biases['decoder_b3']))
	layer4=tf.nn.sigmoid(tf.add(tf.matmul(layer3,weights['decoder_h4']),biases['decoder_b4']))
	return layer4

encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

y_pred=decoder_op
y_true=X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)
	print ("total batch : ",total_batch)

	for epoch in range(training_epochs):
		for i in range(total_batch):
			batch_xs,batch_ys=mnist.train.next_batch(batch_size)
			_,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})

			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

		print("finished")

		encoder_decode=sess.run(y_pred,feed_dict={X:mnist.test.images[:examples_show]})
		f,a=plt.subplots(2,10,figsize=(10,2))
		for i in range(examples_show):
			a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
			a[1][i].imshow(np.reshape(encoder_decode[i], (28, 28)))
		plt.show()
