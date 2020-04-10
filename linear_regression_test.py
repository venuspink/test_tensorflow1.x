
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# df = pd.read_csv('D:/works_tf/test_tensorflow1.x/test_data/Batting.csv')

# print(df.count())

xx_data = [[10.,11.,13.],[21.,22.,25.],[31.,32., 33.],[41.,42.,46.],[52.,53.,59.],
            [61.,63., 69.],[71.,72.,75.],[81.,83., 85.]]
yy_data = [[1.],[2.],[3.],[4.],[5.],[6.],[7.],[8.]]

# xx_data = [[73., 80., 75.],

#           [93., 88., 93.],

#           [89., 91., 90.],

#           [96., 98., 100.],

#           [73., 66., 70.]]

# yy_data = [[1.],

#           [2.],

#           [3.],

#           [4.],

#           [5.]]



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
# x_data = df_XX.values
# y_data = df_HR.values

x_data = np.array(xx_data)
y_data = np.array(yy_data)
print("X_data SHAPE", x_data)
print("Y_data SHAPE", y_data)


X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#선형 회귀분석
hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)


pred = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

print("ACCCCCCC : ",accuracy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

W_history = []
cost_history = []




for step in range(2000):
    cost_val, hy_val, train_val, accr = sess.run(
        [cost, hypothesis, train, pred], feed_dict={X: x_data, Y: y_data})

    # W_history.append(W_val)
    # cost_history.append(cost_val)

    # if step%100 == 0:
        # print(step, "Cost:", cost_val, "accr:",accr, "predict : ",hy_val)

single_test_x = [[202,205,208]]  # 미구엘 사노(미네소타 트윈스)
# single_test_x = min_max_scaler.transform(single_test_x)

single_test_y = [[1000000000000]]
# single_test_y = min_max_scaler_Y.transform(single_test_y)

print("single_test_x", single_test_x, single_test_y)

arr_x = np.array(single_test_x)
arr_y = np.array(single_test_y)



score,acc = sess.run([hypothesis,accuracy], feed_dict={X: arr_x, Y:arr_y})
# score_acc = sess.run(accuracy, feed_dict={X: arr})

print("score : ",score, "acc : ",acc)

# print("예상label ", min_max_scaler_Y.inverse_transform(score))

# answer = [[0]]

# accr = sess.run(accuracy,feed_dict={X:x_data, Y:y_data})
# print("정확도 : ",accr)


# print("정확도 : ", score_acc)

# print(W_history)

# print(accuracy_score(y_data, score))
