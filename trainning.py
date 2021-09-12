import pickle
import numpy as np
import tensorflow as tf
import sys
from tensorflow import keras

n = sys.argv[1]

NFILT = 40

print("Loading training data from pickle")

with open('trainning_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    train_data, train_label = pickle.load(f)

print("Loading validation data from pickle")

with open('validation_data.pkl', 'rb') as f:
    validation_data, validation_label = pickle.load(f)

with open('utterance_list.pkl', 'rb') as f:
    utterance, spk_list, N_SPEAKER = pickle.load(f)


# train model multiple times
print(f"trainning model {n}")
inputs = keras.layers.Input(shape=(NFILT*41,))
dense1 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(inputs)
dense2 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(dense1)
dense3 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(dense2)
drop_out1 = keras.layers.Dropout(0.5)(dense3)
dense4 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(drop_out1)
drop_out2 = keras.layers.Dropout(0.5)(dense4)
outputs = keras.layers.Dense(N_SPEAKER, activation='softmax')(drop_out2)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(np.array(train_data), np.array(train_label), epochs=5, shuffle=True, validation_data=(np.array(validation_data),np.array(validation_label)))
print(f"saving model {n}")
model.save(f"saved_model/{n}")