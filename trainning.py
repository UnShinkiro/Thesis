import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

NFILT = 40

print("Loading training data from pickle")

with open('trainning_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    train_data, train_label = pickle.load(f)

print("Loading validation data from pickle")

with open('validation_data.pkl', 'rb') as f:
    validation_data, validation_label = pickle.load(f)

with open('utterance_list.pkl', 'rb') as f:
    utterance, spk_list, N_SPEAKER = pickle.load(f)

print(len(train_data),len(validation_data))


inputs = keras.layers.Input(shape=(NFILT*41,))
dense1 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(inputs)
dense2 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(dense1)
dense3 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(dense2)
drop_out1 = keras.layers.Dropout(0.5)(dense3)
dense4 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(drop_out1)
drop_out2 = keras.layers.Dropout(0.5)(dense4)
outputs = keras.layers.Dense(N_SPEAKER/2, activation='softmax')(drop_out2)
model = keras.models.Model(inputs=inputs, outputs=outputs)

# train model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(train_data[:len(train_data)/2]), np.array(train_label[:len(train_label)/2]), epochs=10, shuffle=True, validation_data=(np.array(validation_data[:len(validation_data)/2]),np.array(validation_label[:len(validation_label)/2])))
model.save("saved_model/my_model")