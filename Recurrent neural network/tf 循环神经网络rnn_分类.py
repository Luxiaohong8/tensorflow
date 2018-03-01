import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)#设置图级随机seed。
'''
from http://blog.csdn.net/eml_jw/article/details/72353470
让在不同的会话中op产生的随机序列在会话之间是可重复的，相等的
'''
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr=0.001#learning rate
training_iters=100000
batch_size=128#自定义

n_inputs=28 #input的是28行
n_steps=28  #time step
n_hidden_units=128
n_classes=10

x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

#hidden layer
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X,weights,biases):
	#将inputs的shape   X(128batch,28step,28input)==>128 batch*28 steps,28 inputs
	X=tf.reshape(X,[-1,n_inputs])

	#进入hidden层，X_in==（128 batch*28 steps,128 inputs
	X_in = tf.matmul(X, weights['in']) + biases['in']

	# X_in ==> (128 batch, 28 steps, 128 hidden)
	X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

	cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)  
    #n_hidden_units = 128  neurons in hidden layer

    # lstm cell is divided into two parts (c_state, h_state) #batch_size =128
	init_state = cell.zero_state(batch_size, dtype=tf.float32)


	outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
	outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
	results = tf.matmul(outputs[-1], weights['out']) + biases['out'] 
	return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1