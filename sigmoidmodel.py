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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

#데이터 불러오기
data = pd.read_csv('/project2/ppg.csv', encoding='latin1')
data

#데이터 늘리기
data['T1'] = data['PNN50']*data['SDNN']
data['T2'] = data['PNN50']*data['HRV']
data['T3'] = data['PNN50']*data['LF']
data['T4'] = data['PNN50']*data['HR']
data['T5'] = data['SDNN']*data['PNN50']*data['LF']
data['T6'] = data['SDNN']*data['LF']
data['T7'] = data['SDNN']*data['HRV']
data['T8'] = data['SDNN']*data['HR']
data['T9'] = data['HRV']*data['PNN50']*data['SDNN']
data['T10'] = data['HRV']*data['LF']*data['SDNN']
data['T11'] = data['HRV']*data['HR']
data['T12'] = data['HRV']*data['LF']
data['T13'] = data['LF']*data['PNN50']*data['SDNN']
data['T14'] = data['LF']*data['SDNN']*data['HRV']
data['T15'] = data['LF']*data['HRV']*data['HR']
data['T16'] = data['LF']*data['HR']
data['T17'] = data['HR']*data['PNN50']*data['SDNN']
data['T18'] = data['HR']*data['SDNN']*data['HRV']
data['T19'] = data['HR']*data['HRV']*data['LF']
data['T20'] = data['HR']*data['PNN50']*data['HRV']
data

#상관 관계도 분석
data_corr = data.corr()
plt.figure(figsize=(15,15))
sns.set(font_scale=0.8)
sns.heatmap(data_corr, annot=True, cbar=False)
plt.show()


#데이터 분리 및 스케일링
blood_sugar_rate = data
x = blood_sugar_rate[['HR', 'HRV','SDNN','RMSSD','PNN50','VLF','LF','HF', 'gender','age','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15','T16','T17','T18','T19','T20']]              # x축에 input 데이터 나열
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

#모델
from keras.callbacks import learning_rate_schedule
model = Sequential()
model.add(Dense(32, input_dim=29, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#모델 학습
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model_2.h5',   #ModelCheckpoint의 객체 #'best-model.h5'이름으로 저장
                                                save_best_only=True) #모델이 "최상"으로 간주될 때만 저장
#early stopping
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, #EarlyStopping 객체
                                                  restore_best_weights=True) #이상한 패턴이 2번 이상 지속되면 stop
                                                  #restore_best_weights=True : 가장 낮은 검증 손실을 낸 모델을 파일에 저장
history = model.fit(x_train_scaled, y_train
                , epochs = 100
                , batch_size = 64
                , validation_data=(x_val_scaled, y_val_scaled)
                , callbacks=[checkpoint_cb, early_stopping_cb] #best 모델 저장 및 early stopping
                )

#학습된 모델 시각화
plt.plot(history.history['accuracy'], 'r'),
plt.plot(history.history['val_accuracy'], 'g'),
plt.plot(history.history['loss'], 'b'),
plt.plot(history.history['val_loss'], 'y')