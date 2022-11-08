from tensorflow import keras
from tensorflow.keras import Sequential
from keras import layers
from tensorflow.keras .layers import Dense, Flatten, Dropout, Conv2D, Maxpool2D, BatchNormalization
from tensorflow.keras.allbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

m = Sequential()
m.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
m.add(Dropout(0.3))
m.add(BatchNormalization())
m.add(layers.Dense(16, activation='relu'))
m.add(Dropout(0.3))
m.add(BatchNormalization())
m.add(layers.Dense(16, activation='relu'))
m.add(Dropout(0.3))
m.add(BatchNormalization())
m.add(layers.Dense(16, activation='relu'))
m.add(Dropout(0.3))
m.add(BatchNormalization())
m.add(layers.Dense(16, activation='relu'))
m.add(Dropout(0.3))
m.add(BatchNormalization())
m.add(layers.Dense(16, activation='relu'))
m.add(Dropout(0.3))
m.add(BatchNormalization())
m.add(layers.Dense(16, activation='relu'))
m.add(Dropout(0.3))
m.add(BatchNormalization())
m.add(layers.Dense(1, activation='sigmoid'))

m.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuacy'])
his = m.fit()