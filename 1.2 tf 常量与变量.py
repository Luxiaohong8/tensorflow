import tensorflow as tf

#定义一个变量
var=tf.Variable(0,name="myvar")

#定义一个常量
con_var=tf.constant(1)

new_var=tf.add(var,con_var)


init =tf.global_variables_initializer()

with tf.Session()as sess:
	sess.run(init)
	print(sess.run(var),sess.run(con_var),sess.run(new_var))


# placeholder 是 Tensorflow 中的占位符
# 如果想要从外部传入data, 那就需要用到 tf.placeholder()
# 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))