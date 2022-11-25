from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization, Flatten

data = pd.read_csv('/project2/ppg.csv', encoding='latin1') 
data

blood_sugar_rate_lr = data
x_lr = blood_sugar_rate_lr[['HR', 'HRV','SDNN','RMSSD','PNN50','VLF','LF','HF','gender','age']]              # x축에 input 데이터 나열
y_lr = blood_sugar_rate_lr[['blood_sugar']]              # y축에 타겟 데이터 나열

x_train_lr, x_test_lr, y_train_lr, y_test_lr = \
  train_test_split(x_lr, y_lr, stratify=y_lr, test_size=0.2, random_state=42)      # 훈련 데이터와 테스트 데이터 분류

x_train_lr1, x_val_lr, y_train_lr1, y_val_lr = \
  train_test_split(x_train_lr, y_train_lr, stratify=y_train_lr, \
                   test_size=0.2, random_state=42)                     # 훈련 데이터와 검증 데이터 분류          
 
scaler_lr = StandardScaler()   # 객체 만들기
scaler_lr.fit(x_train_lr1)     # 변환 규칙을 익히기
x_train_scaled_lr = scaler_lr.transform(x_train_lr1)  # 데이터를 표준화 전처리
x_test_scaled_lr = scaler_lr.transform(x_test_lr)
x_val_scaled_lr = scaler_lr.transform(x_val_lr)      # 데이터를 표준화 전처리

from sklearn.metrics import r2_score, mean_squared_error
import joblib

poly = PolynomialFeatures(include_bias=False, degree=2)

x_train_poly = poly.fit_transform(x_train_scaled_lr)
x_test_poly = poly.fit_transform(x_test_scaled_lr)

lr = LinearRegression().fit(x_train_poly, y_train_lr1)

y_pred_train = lr.predict(x_train_poly)
y_pred_test = lr.predict(x_test_poly)

print(f"다항회귀 적용 : {r2_score(y_train_lr1, y_pred_train)}, {r2_score(y_test_lr, y_pred_test)}")
print(f"평균 제곱 오차 : {np.sqrt(mean_squared_error(y_train_lr1, y_pred_train))}, {np.sqrt(mean_squared_error(y_test_lr, y_pred_test))}")

joblib.dump(lr, '/content/blood_sugar.pkl')

#feature 늘리기 과제