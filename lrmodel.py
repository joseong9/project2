import pandas as pd
data = pd.read_csv('/content/ppg.csv', encoding='latin1')
data

blood_sugar_rate2 = data
x2 = blood_sugar_rate2[['HR', 'HRV','SDNN','RMSSD','PNN50','VLF','LF','HF','gender','age']]              # x축에 input 데이터 나열
y2 = blood_sugar_rate2[['blood_sugar']]              # y축에 타겟 데이터 나열

x_train_all2, x_test2, y_train_all2, y_test2 = \
  train_test_split(x2, y2, stratify=y2, test_size=0.2, random_state=42)      # 훈련 데이터와 테스트 데이터 분류

x_train2, x_val2, y_train2, y_val2 = \
  train_test_split(x_train_all2, y_train_all2, stratify=y_train_all2, \
                   test_size=0.2, random_state=42)                     # 훈련 데이터와 검증 데이터 분류          
 
scaler2 = StandardScaler()   # 객체 만들기
scaler2.fit(x_train2)     # 변환 규칙을 익히기
x_train_scaled2 = scaler.transform(x_train2)  # 데이터를 표준화 전처리
x_test_scaled2 = scaler.transform(x_test2)
y_train_scaled2 = y_train2.values
y_test_scaled2 = y_test2.values
x_val_scaled2 = scaler.transform(x_val2)      # 데이터를 표준화 전처리
y_val_scaled2 = y_val2.values


from sklearn.metrics import r2_score
poly = PolynomialFeatures(include_bias=False, degree=2)
x_train2_poly = poly.fit_transform(x_train2)
x_test2_poly = poly.fit_transform(x_test2)

lr = LinearRegression().fit(x_train2, y_train2)
y_pred = lr.predict(x_test2)
print(f"다항회귀 미적용 : {r2_score(y_test2, y_pred)}")

lr2 = LinearRegression().fit(x_train2_poly, y_train2)
y_pred2 = lr2.predict(x_test2_poly)
print(f"다항회귀 적용 : {r2_score(y_test2, y_pred)}")
