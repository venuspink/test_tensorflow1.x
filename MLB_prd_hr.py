"""
MLB타자들의 데이터를 머신러닝하여
입력되는 스탯에 따라 삼진개수를 예측해본다.

"""

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('D:/works_tf/test_tensorflow1.x/test_data/Batting.csv')

# print(df.count())

#1980년 이후 데이터로 추리고.. 300타수 이상
df = df[df['yearID'] >= 1960]
df = df[df['AB'] >= 200]
# 타점 또는 홈런이 0점이면 제외
df = df[(df.RBI > 0) & (df.HR > 0)]

#특정 인물 ID 찾아서 행삭제하기
del_target_idx = df[df.playerID == 'sanomi01'].index
print("삭제 타겟 index : ", del_target_idx)
df = df.drop(del_target_idx)


#출루율 (안타 + 사사구) / (타수 + 사사구 + 희생플라이)
df['OBP'] = (df['H']+df['2B']+df['3B']+df['HR']+df['BB']+df['HBP'])/(df['AB']+df['BB']+df['HBP']+df['SH']+df['SF'])
#장타율 [단타 + (2*2루타) + (3*3루타) + (4*홈런] / 타수
df['SLG'] = (df['H']+(2*df['2B'])+(3*df['3B'])+(4*df['HR']))/df['AB']

df['OBP'] = df.OBP.round(3)
df['SLG'] = df.SLG.round(3)

#OPS 장타율 + 출루율
df['OPS'] = df['OBP'] + df['SLG']

# 특정 컬럼만으로 정제
#  'AB', 'H', '2B','HR', 'BB'
# df_XX = df.loc[:,['AB','H','2B','HR','BB','OBP','SLG','OPS']]
df_XX = df.loc[:,['AB','HR','BB','OBP']]
df_HR = df.loc[:, ['SO']]



print(df.head())


df_XX = df_XX.reset_index(drop=True)
df_HR = df_HR.reset_index(drop=True)

print("입력변수 : ",df_XX.count())
print("종속변수 : ",df_HR.count())

# print(len(train), '훈련샘플')
# print(len(val), '검증샘플')
# print(len(test), '테스트샘플')

#데이터 표준화 Normalization
x_temp = df_XX.values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(x_temp)
# x_scaled = min_max_scaler.fit_transform(x_temp)
x_scaled = min_max_scaler.transform(x_temp)

df_XX = pd.DataFrame(x_scaled, columns=df_XX.columns)

y_temp = df_HR.values.astype(float)
min_max_scaler_Y = preprocessing.MinMaxScaler()
min_max_scaler_Y.fit(y_temp)
y_scaled = min_max_scaler_Y.transform(y_temp)

df_HR = pd.DataFrame(y_scaled, columns=df_HR.columns)

# y_temp = df_HR.values.astype(float)
# y_scaled = min_max_scaler.fit_transform(y_temp)
# df_HR = pd.DataFrame(y_scaled, columns=df_HR.columns)
# print(df_XX.head())
# print(df_HR.head())

# df_HR = df_XX['HR']
# df_HR = df_XX.loc[:, ['HR']]
# df_XX = df_XX.drop(['HR'], axis=1)
# print(df_HR.head())

# 훈련, 테스트 데이터 분할
# train_X, test_X = train_test_split(df_XX, test_size=0.2)
# train_Y, test_Y = train_test_split(df_HR, test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)

# 훈련 입력값 정의
x_data = df_XX.values
y_data = df_HR.values

print("X_data SHAPE", x_data.shape)
print("Y_data SHAPE", y_data.shape)

# x_test_data = test_X.values


X = tf.placeholder(tf.float32, shape=[None,4])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([4,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#선형 회귀분석
hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(10000):
    cost_val, train_val, acc = sess.run(
        [cost, train, accuracy], feed_dict={X: x_data, Y: y_data})

    if step%100 == 0:
        print(step, "Cost:",cost_val, "ACC : ",acc )


#  'AB', 'H', '2B','HR', 'BB'
# single_test_x = [[489,161,29,44,80]] #옐리치
# single_test_x = [[470,137,27,2,45,104,110]] #트라웃
# single_test_x = [[570,140,28,2,31,87,50]] #업튼
# single_test_x = [[345,89,24,8,18]] #스즈키

single_test_x = [[279, 18, 53, 0.495]]  # 미구엘 사노(미네소타 트윈스)
single_test_x = min_max_scaler.transform(single_test_x)

print("single_test_x", single_test_x)

arr = np.array(single_test_x)

score = sess.run(hypothesis, feed_dict={X: arr})

print("예상 삼진개수 ", min_max_scaler_Y.inverse_transform(score))

# answer = [[119]]
# accr = sess.run(accuracy, feed_dict={X: arr, Y: answer})
# print("정확도 : ", accr)

