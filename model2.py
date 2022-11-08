import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def create_model():
    m = keras.sequential()
    m.add(keras.layers.Flatten(input_shape=(28,28)))
    m.add(keras.layers.Dense(256, activation='relu'))
    m.add(keras.layers.Dense(128, activation='relu'))
    m.add(keras.layers.Dense(64, activation='relu'))
    
    m.add(keras.laters.Dense(4, activation='softmax'))
    
    return m

model = create_model()

model.compile(optimizer=tf.keras.optimers.Adam(0.01), loss='categorical_crossentropy', metrics=['acc'])

his = model.fit(train_dataset, epochs=100, steps_per_epoch=3, validation_data=test_dataset, validation_steps=3)

