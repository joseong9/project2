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


from keras.callbacks import learning_rate_schedule
model = Sequential()
model.add(Dense(32, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

m.fit(x_train_scaled, y_train_scaled
      , epochs = 100
      , batch_size = 64
      , validation_data=(x_val_scaled, y_val_scaled)
      )

mlp = MLPClassifier(
                    hidden_layer_sizes=(3417,),        # tuple로 전달할 수 있으며, tuple의 i번째 element가 i번째 hidden layer의 크기
                    activation='relu',              # hidden layer에 사용할 activation function의 종류 
                    solver='sgd',                   # optimizer의 종류 lbfgs, sgd, adam
                    alpha=0.001,
                    batch_size='auto',              # mini-batch의 크기를 설정한다
                    learning_rate='constant',       # learning rate scheduler의 종류 설정
                    learning_rate_init=0.001,        
                    max_iter=100,                   # training iteration을 수행할 횟수(epoch)
                    )
mlp.fit(x_train_scaled, y_train)
mlp.score(x_val_scaled, y_val)

scores_train = m.evaluate(x_test,y_test)
scores_val = m.evaluate(x_val_scaled,y_val_scaled)
print("%s: %.2f%%,  val_losss: %.2f%%" %(m.metrics_names[0], scores_train[1]*100, scores_val[1]*100))

