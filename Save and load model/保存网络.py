from __future__ import print_function
import tensorflow as tf
import numpy as np

W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weight')
b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

init=tf.global_variables_initializer()

saver=tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	save_path=saver.save(sess,"./my_net/test1.ckpt")
	print(save_path)