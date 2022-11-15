from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization, Flatten
import pandas as pd
data = pd.read_csv('/content/ppg.csv', encoding='latin1')
data

blood_sugar_rate = data
x = blood_sugar_rate[['HR','HRV','SDNN','RMSSD','PNN50','VLF','LF','HF','gender','age']]                # x축에 input 데이터 나열
y = blood_sugar_rate[['blood_sugar']]              # y축에 타겟 데이터 나열
y = y >= 126
y["blood_sugar"] = np.where(y["blood_sugar"]==True, np.int8(1), y["blood_sugar"])
y["blood_sugar"] = np.where(y["blood_sugar"]==False, np.int8(0), y["blood_sugar"])

x_train_all, x_test, y_train_all, y_test = \
  train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)      # 훈련 데이터와 테스트 데이터 분류

x_train, x_val, y_train, y_val = \
  train_test_split(x_train_all,y_train_all,stratify=y_train_all, \
                   test_size=0.2,random_state=42)                     # 훈련 데이터와 검증 데이터 분류          
 
scaler = StandardScaler()   # 객체 만들기
scaler.fit(x_train)     # 변환 규칙을 익히기
x_train_scaled = scaler.transform(x_train)  # 데이터를 표준화 전처리
x_test_scaled = scaler.transform(x_test)
y_train_scaled = y_train.values
y_test_scaled = y_test.values
x_val_scaled = scaler.transform(x_val)      # 데이터를 표준화 전처리
y_val_scaled = y_val.values

mlp = MLPClassifier(
                    hidden_layer_sizes=(9,),        # tuple로 전달할 수 있으며, tuple의 i번째 element가 i번째 hidden layer의 크기
                    activation='relu',              # hidden layer에 사용할 activation function의 종류 
                    solver='sgd',                   # optimizer의 종류 lbfgs, sgd, adam
                    alpha=0.001,
                    batch_size='auto',              # mini-batch의 크기를 설정한다
                    learning_rate='constant',       # learning rate scheduler의 종류 설정
                    learning_rate_init=0.001,        
                    max_iter=500,                   # training iteration을 수행할 횟수(epoch)
                    )
mlp.fit(x_train_scaled, y_train)
mlp.score(x_val_scaled, y_val)

mlp.score(x_train_scaled,y_train)
mlp.score(x_test_scaled, y_test_scaled)