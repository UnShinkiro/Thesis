from scipy.io import wavfile
import os
import pickle
import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from pre_process import form_input_data

INT16_MAX = 32767
SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * 0.025)
NFFT = 512 
NFILT = 40
N_SPEAKER = 32
pre_emphasis = 0.97

spk_list = os.listdir("vox/vox1_dev_wav")
utterance = {}
emphasized_data = []
validation_dataset = []
validation_data = []
validation_label = []
train_data = []
train_label = []
enrollment_dataset = []
verification_dataset = []

# Text-independent Data processing
for pid, speaker in enumerate(spk_list[0:N_SPEAKER]):
    utterance[speaker] = {}
    path = "vox/vox1_dev_wav/" + speaker
    folders = os.listdir(path)
    utterance[speaker]['files'] = []
    for folder in folders:
        if folder[0] == ".":
            pass
        else:
            path = "vox/vox1_dev_wav/" + speaker + "/" + folder
            try:
                files = os.listdir(path)
                for file in files:
                    utterance[speaker]['files'].append(folder + "/" + file)
            except:
                pass
        for count in range(10):
            file_path = "vox/vox1_dev_wav/" + speaker + "/" + utterance[speaker]['files'].pop(0)
            try:
                _, data = wavfile.read(file_path)         # requires tons of memory with many spekaers
                emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
                if count < 5:
                    emphasized_data.append((emphasized_signal,pid))
                elif count < 10:
                    validation_dataset.append((emphasized_signal,pid))
            except:
                pass

for entry in emphasized_data:
    form_input_data(entry, train_data, train_label)
for entry in validation_dataset:
    form_input_data(entry, validation_data, validation_label)

with open('trainning_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([train_data, train_label, validation_data, validation_label], f)
with open('utterance_list.pkl', 'wb') as f:
    pickle.dump([utterance, spk_list], f)

inputs = keras.layers.Input(shape=(NFILT*41,))
dense1 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(inputs)
dense2 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(dense1)
dense3 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(dense2)
drop_out1 = keras.layers.Dropout(0.5)(dense3)
dense4 = keras.layers.Dense(256, kernel_regularizer='l2', activation='relu')(drop_out1)
drop_out2 = keras.layers.Dropout(0.5)(dense4)
outputs = keras.layers.Dense(N_SPEAKER, activation='softmax')(drop_out2)
model = keras.models.Model(inputs=inputs, outputs=outputs)

# train model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(train_data), np.array(train_label), epochs=50, shuffle=True, validation_data=(np.array(validation_data),np.array(validation_label)))
model.save("saved_model/my_model")