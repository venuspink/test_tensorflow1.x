"""
MLB타자들의 데이터를 머신러닝하여
입력되는 데이터가 홈런타자가 될 가능성이 있는지 예측한다.

"""

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('D:/works_tf/test_tensorflow1.x/test_data/Batting.csv')

# print(df.count())

#1980년 이후 데이터로 추리고..
df = df[df['yearID'] >= 1980]
# 타점 또는 홈런이 0점이면 제외
df = df[(df.RBI > 0) & (df.HR > 0)]
# 특정 컬럼만으로 정제
df_XX = df.loc[:,['G','AB','R','H','2B','3B','RBI','BB','SO']]
df_HR = df.loc[:, ['HR']]

df_XX = df_XX.reset_index(drop=True)
df_HR = df_HR.reset_index(drop=True)

# print(len(train), '훈련샘플')
# print(len(val), '검증샘플')
# print(len(test), '테스트샘플')

#데이터 표준화 Normalization
x_temp = df_XX.values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x_temp)
df_XX = pd.DataFrame(x_scaled, columns=df_XX.columns)
# print(df.head())

# 훈련, 테스트 데이터 분할
train, test = train_test_split(df_XX, test_size=0.2)
train_Y, test_Y = train_test_split(df_HR, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)

# 훈련 입력값 정의
x_data = train.values
y_data = train_Y.values

x_test_data = test.values


X = tf.placeholder(tf.float32, shape=[None,9])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([9,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#선형 회귀분석
hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(cost)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(1000):
    cost_val, train_val = sess.run([cost,train], feed_dict = {X:x_data, Y:y_data})

    if step%100 == 0:
        print(step, "Cost:",cost_val )


print("H예측 : ", sess.run(hypothesis, feed_dict={X:x_test_data}))


