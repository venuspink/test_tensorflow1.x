import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist는 훈련(training), 테스트(testing) 그리고 검증(validation) 데이터를 NumPy 배열로 저장하는 클래스입니다

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

# 비용최소화 학습
train = tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

# 예측값의 정의 : 가설(hypothesis)과 실제계산 값(y_)의 비교하여 예측값 도출
# argmax(hypothesis, 1) : 모델이 입력을 받고 가장 그럴듯하다고 생각한 레이블
# argmax(y_, 1) : 실제 레이블
# return : boolean[]
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y_, 1))
print("correct_prediction : ", correct_prediction)

# 정확도의 정의 : 예측값의 평균
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("accuracy : ", accuracy)

# 테스트데이터를 대상으로 정확도 계산
print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))
