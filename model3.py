import pandas as pd
data = pd.read_csv('./project2/project/ppg.csv', encoding='latin1')
data

data.isnull().sum()

import numpy as np
len(np.unique(data.HF,return_counts=True)[1])


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
 
blood_sugar_rate = data
x=blood_sugar_rate[['HR']]              # x축에 input 데이터 나열
y=blood_sugar_rate[['blood_sugar']]              # y축에 타겟 데이터 나열
y = y >= 126
# y.loc[y["blood_sugar"] == True, "blood_sugar"] = np.int64(1)
# y.loc[y["blood_sugar"] == False, "blood_sugar"] = np.int64(0)
y["blood_sugar"] = np.where(y["blood_sugar"]==True, np.int8(1), y["blood_sugar"])
y["blood_sugar"] = np.where(y["blood_sugar"]==False, np.int8(0), y["blood_sugar"])

x_train_all, x_test, y_train_all, y_test = \
  train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)  # 훈련 데이터와 테스트 데이터 분류
x_train, x_val, y_train, y_val = \
  train_test_split(x_train_all,y_train_all,stratify=y_train_all, \
                   test_size=0.2,random_state=42)  # 훈련 데이터와 검증 데이터 분류
 
scaler = StandardScaler()   # 객체 만들기
scaler.fit(x_train)     # 변환 규칙을 익히기
x_train_scaled = scaler.transform(x_train)  # 데이터를 표준화 전처리
# y_train_scaled = scaler.transform(y_train)
y_train_scaled = y_train.values
x_val_scaled = scaler.transform(x_val)      # 데이터를 표준화 전처리
# y_val_scaled = scaler.transform(y_val)
y_val_scaled = y_val.values


 
m = Sequential()
m.add(Input(shape=(1)))
m.add(Dense(4, activation='relu'))
m.add(Dense(8, activation = 'relu'))
m.add(Dense(16, activation = 'relu'))
m.add(Dense(8, activation = 'relu'))
m.add(Dense(1, activation = 'sigmoid'))

m.summary()

m.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

m.fit(x_train_scaled, y_train_scaled
      , epochs = 100
      , batch_size = 64
      , validation_data=(x_val_scaled, y_val_scaled)
      )

scores_train = m.evaluate(x_test,y_test)
scores_val = m.evaluate(x_val_scaled,y_val_scaled)
print("%s: %.2f%%,  val_losss: %.2f%%" %(m.metrics_names[0], scores_train[1]*100, scores_val[1]*100))

