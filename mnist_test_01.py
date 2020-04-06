import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 처음 작성 후, 실행 (ctrl+F5) 하면 MNIST_data폴더가 생성되고 테스트파일이 자동 다운된다.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 가설
hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

# cost
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(hypothesis), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))
