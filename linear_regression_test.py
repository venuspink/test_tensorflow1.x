
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# df = pd.read_csv('D:/works_tf/test_tensorflow1.x/test_data/Batting.csv')

# print(df.count())

xx_data = [[10,10],[20,20],[30,30],[40,40],[50,50],[60,60],[70,70],[80,80]]
yy_data = [[1],[2],[3],[4],[5],[6],[7],[8]]


#데이터 표준화 Normalization
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(xx_data)
# x_scaled = min_max_scaler.fit_transform(x_temp)
x_scaled = min_max_scaler.transform(xx_data)

df_XX = pd.DataFrame(x_scaled)

min_max_scaler_Y = preprocessing.MinMaxScaler()
min_max_scaler_Y.fit(yy_data)
y_scaled = min_max_scaler_Y.transform(yy_data)

df_HR = pd.DataFrame(y_scaled)

# 훈련 입력값 정의
x_data = df_XX.values
y_data = df_HR.values

print("X_data SHAPE", x_data.shape)
print("Y_data SHAPE", y_data.shape)


X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#선형 회귀분석
hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


pred = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

W_history = []
cost_history = []




for step in range(1000):
    cost_val, train_val, acc = sess.run(
        [cost, train, accuracy], feed_dict={X: x_data, Y: y_data})

    # W_history.append(W_val)
    cost_history.append(cost_val)

    if step%100 == 0:
        print(step, "Cost:",cost_val ,"ACCURACY : ",acc)

single_test_x = [[150,150]]  # 미구엘 사노(미네소타 트윈스)
single_test_x = min_max_scaler.transform(single_test_x)

print("single_test_x", single_test_x)

arr = np.array(single_test_x)


score = sess.run(hypothesis, feed_dict={X: arr})
# score_acc = sess.run(accuracy, feed_dict={X: arr})

print("예상 삼진개수 ", min_max_scaler_Y.inverse_transform(score))

answer = [[0]]
accr = sess.run(accuracy,feed_dict={X:arr, Y:answer})
print("정확도 : ",accr)


# print("정확도 : ", score_acc)

# print(W_history)

# print(accuracy_score(y_data, score))
