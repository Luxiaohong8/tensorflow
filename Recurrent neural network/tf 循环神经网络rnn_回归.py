import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters
TIME_STEP = 10       # rnn time step
INPUT_SIZE = 1      # rnn input size
CELL_SIZE = 32      # rnn cell size
LR = 0.02           # learning rate

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)    # float32 for converting torch FloatTensor
plt.plot(steps, y_np, 'r-', label='target (cos)') 
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # shape(batch, 5, 1)
tf_y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])          # input y

rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE)
init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, tf_x, initial_state=init_state, time_major=False)

outs2D=tf.reshape(outputs,[-1,CELL_SIZE])
net_outs2D = tf.contrib.layers.fully_connected(outs2D, INPUT_SIZE)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE]) 

loss = tf.contrib.losses.mean_squared_error(labels=tf_y, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)


with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	plt.figure(1, figsize=(12, 5))
	plt.ion() 
	for step in range(1000):
		start, end = step * np.pi, (step+1)*np.pi
		steps = np.linspace(start, end, TIME_STEP)
		x = np.sin(steps)[np.newaxis, :, np.newaxis]    # shape (batch, time_step, input_size)
		y = np.cos(steps)[np.newaxis, :, np.newaxis]
		if 'final_s_' not in globals():
			feed_dict = {tf_x: x, tf_y: y}
		else:                                           # has hidden state, so pass it to rnn
			feed_dict = {tf_x: x, tf_y: y, init_state: final_s_}
		_, pred_, final_s_ = sess.run([train_op, outs, final_state], feed_dict)     # train
	# plotting
	plt.plot(steps, y.flatten(), 'r-')
	plt.plot(steps, pred_.flatten(), 'b-')
	plt.ylim((-1.2, 1.2)); plt.draw()
	plt.pause(0.05)
plt.ioff()
plt.show()