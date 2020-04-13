import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('D:/works_tf/test_tensorflow1.x/test_data/Teams.csv')

df = df[df['yearID'] >= 1970]

#출루율 (안타 + 사사구) / (타수 + 사사구 + 희생플라이)
df['OBP'] = (df['H']+df['2B']+df['3B']+df['HR']+df['BB']+df['HBP'])/(df['AB']+df['BB']+df['HBP']+df['SF'])

#장타율 [단타 + (2*2루타) + (3*3루타) + (4*홈런] / 타수
df['SLG'] = (df['H']+(2*df['2B'])+(3*df['3B'])+(4*df['HR']))/df['AB']

df['OBP'] = df.OBP.round(3)
df['SLG'] = df.SLG.round(3)

x_data = df.loc[:, ['OBP', 'SLG']]
y_data = df.loc[:, ['R']]
y_data = y_data / 1000

x_data.reset_index(drop=True)

print(y_data.head())

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#선형 회귀분석
hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(1000):
    cost_val, hy_val, train_val, w_val,b_val = sess.run(
        [cost, hypothesis,train,W,b], feed_dict={X: x_data, Y: y_data})

    if step % 100 == 0:
        print(step, "Cost:", cost_val, "b_val",b_val)


t_xx = [[0.519,0.625]]

print(sess.run(hypothesis, feed_dict={X:t_xx}))


plt.figure(figsize=(10, 8))
plt.scatter(x_data.OBP, y_data * 1000, c="red")
plt.scatter(x_data.SLG, y_data * 1000, c="blue")
# plt.plot(x, abline_values, 'b')
# plt.title("Slope = %s" % (slope))
plt.xlabel("obp")
plt.ylabel("Run score")
plt.show()
