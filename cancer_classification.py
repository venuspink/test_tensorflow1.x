from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

# 유전자 발현에 근거한 암 분류 모델 데이터
# Cancer_1 -> 1(암환자), 0(정상인)
xy = np.loadtxt('D:/works_tf/test_tensorflow1.x/test_data/cancer_data.csv',delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, -1:]

print(y_data)

# 프로그램 실행 순간에 변수값을 입력하기 위해 placeholder 함수 사용
X = tf.placeholder(tf.float32, shape=[None, 10])  # 10개의 feature로 구성 된 shape.
Y = tf.placeholder(tf.float32, shape=[None, 1])  # 1개의 결과로 구성된 shape

# W와 b값의 초기값 정보가 없기에 랜덤하게 값을 설정
W = tf.Variable(tf.random_normal([10, 1], mean=0.01, stddev=0.01), name='weight')  # 10개가 입력되서 1개의 결과가를 가짐
b = tf.Variable(tf.random_normal([1]))

# Logistic Regression을 적용(tf.sigmoid() == 0과 1 사이의 값을 만들기 위한 시그모이드 함수)
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost = tf.reduce_mean(tf.square(hypothesis - Y))
# cost = tf.reduce_mean(-Y * tf.log(hypothesis ) + (1-Y) * tf.log(1-hypothesis ))
cost = -tf.reduce_mean(Y * tf.log(hypothesis ) + (1 - Y) * tf.log(1 - hypothesis))

# lerning_rate 가 중요함. 최초 0.01부터 시작해서 조절하면 됨.
train = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    # if step%1000 == 0:
    #     print(step, "COST : ", cost_val, "hypothesis : ",hy_val)


h, c, a = sess.run([hypothesis,predict, accuracy], feed_dict={X:x_data, Y:y_data})


# print("\nHypothesis : ",h, "\nCorrect(Y) :", c, "\nAccuracy:",a)


for index, value in enumerate(h):
    if index%100 == 0:
        print("INDEX : ",index , "예측값 :", value, "암환자여부 : ",'정상' if c[index] == 0 else  '암환자')




