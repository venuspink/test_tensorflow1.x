import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost import XGBRegressor
import numpy as np

df = pd.read_csv('D:/works_tf/test_tensorflow1.x/test_data/Teams.csv')

df = df[df['yearID'] >= 1970]

#출루율 (안타 + 사사구) / (타수 + 사사구 + 희생플라이)
df['OBP'] = (df['H']+df['2B']+df['3B']+df['HR']+df['BB'] + df['HBP'])/(df['AB']+df['BB']+df['HBP']+df['SF'])

#장타율 [단타 + (2*2루타) + (3*3루타) + (4*홈런] / 타수
df['SLG'] = (df['H']+(2*df['2B'])+(3*df['3B'])+(4*df['HR']))/df['AB']

df['OBP'] = df.OBP.round(3)
df['SLG'] = df.SLG.round(3)

x_data = df.loc[:, ['OBP', 'SLG']]
y_data = df.loc[:, ['R']]
y_data = y_data / 1000

x_data.reset_index(drop=True)
print(y_data.head())

my_model = XGBRegressor()
my_model.fit(x_data.values, y_data.values,verbose=False)


test_x = np.array([[0.325,0.415],[0.256,0.333]], dtype=float)
predictions = my_model.predict(test_x)

print(predictions)
