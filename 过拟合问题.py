import tensorflow as tf
import numpy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#加载数据
digits=load_digits()
x=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

def add_layer(inputs,input_size,output_size,layer_name,activation_function=None):
	Weigths=tf.Variable(tf.random_normal([input_size,output_size]))
	#在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
	biases=tf.Variable(tf.zeros([1,output_size])+0.1)
	#未激活的值
	y=tf.matmul(inputs,Weigths)+biases

	#此处使用dropout的地方，丢掉一部分y
	y=tf.nn.dropout(y,keep_prob)
	if activation_function is None:
		outputs=y
	else:
		outputs=activation_function(y)
	tf.summary.histogram(layer_name + '/outputs', outputs)
	return outputs

keep_prob=tf.placeholder(tf.float32)

xs=tf.placeholder(tf.float32,[None,64])#8x8
ys=tf.placeholder(tf.float32,[None,10])#十个类别

layer1=add_layer(xs,64,50,'layer1',activation_function=tf.nn.tanh)
prediction=add_layer(layer1,50,10,'layer2',activation_function=tf.nn.softmax)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	merged=tf.summary.merge_all()
	train_writer=tf.summary.FileWriter("logs/train",sess.graph)
	test_writer=tf.summary.FileWriter("logs/test",sess.graph)
	sess.run(init)

	for i in range(500):
		sess.run(train_step,feed_dict={xs:x_train,ys:y_train,keep_prob:0.5})
		if i%50==0:
			train_result=sess.run(merged,feed_dict={xs:x_train,ys:y_train,keep_prob:1})
			test_result=sess.run(merged,feed_dict={xs:x_test,ys:y_test,keep_prob:1})
			train_writer.add_summary(train_result,i)
			test_writer.add_summary(test_result,i)



