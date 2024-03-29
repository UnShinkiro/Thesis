from scipy.io import wavfile
import os
import sys
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pre_process import form_input_data

N_MODEL = 20
N_UTTERANCE = int(sys.argv[1])
for n in range(N_MODEL):
    os.makedirs(f"d-vector/{N_UTTERANCE}/{n}", exist_ok=True)
pre_emphasis = 0.97
intermediate_layer_model = []

for n in range(N_MODEL):
    model = tf.keras.models.load_model(f"saved_model/{n}")
    #model.summary()
    layer_name = 'dropout_1'
    intermediate_layer_model.append(keras.models.Model(inputs=model.input,
                                                    outputs=model.get_layer(layer_name).output))

spk_list = os.listdir("vox/vox1_test_wav")
utterance = {}
n = 0
while n in range(len(spk_list)):
    if spk_list[n].startswith("."):
        spk_list.pop(n)
        n -= 1
    n += 1

for pid, speaker in enumerate(spk_list):
    print(f"Logging speaker {pid}")
    utterance[speaker] = {}
    path = "vox/vox1_test_wav/" + speaker
    folders = os.listdir(path)
    utterance[speaker]['files'] = []
    for folder in folders:
        if not folder.startswith("."):
            path = "vox/vox1_test_wav/" + speaker + "/" + folder
            files = os.listdir(path)
            for file in files:
                if not file.startswith("."):
                    utterance[speaker]['files'].append(folder + "/" + file)

for speaker in spk_list:
    print(f"enrolling speaker {speaker}")
    enrollment_dataset = []
    for count in range(30):
        file_path = "vox/vox1_test_wav/" + speaker + "/" + \
            utterance[speaker]['files'].pop(0)
        if count < N_UTTERANCE:
            _, data = wavfile.read(file_path)         # requires tons of memory with many spekaers
            emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
            enrollment_dataset.append((emphasized_signal,pid))

    enrollment_data = []
    enrollment_label = [] #Not used
    d_utterance_list = []
    
    for n in range(N_MODEL):
        print(f'using model {n} to enrol speaker {speaker}')
        d_utterance_list.clear()
        for entry in enrollment_dataset:
            enrollment_data.clear()
            enrollment_label.clear()
            form_input_data(entry, enrollment_data, enrollment_label)
            intermediate_output = intermediate_layer_model[n].predict(np.array(enrollment_data))
            d_utterance = np.zeros(256)
            for out in intermediate_output:
                d_utterance += out/sum(out)
            d_utterance_list.append(d_utterance) # Saving the utterance d-vector for future uncertainty measure
        d_model = np.zeros(256)
        for vector in d_utterance_list:
            d_model += vector
        d_model = d_model/len(d_utterance_list)
        filename = f'd-vector/{N_UTTERANCE}/{n}/' + speaker + '.pkl'
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([d_utterance_list, d_model], f)

with open('test_utterance.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([utterance, spk_list], f)

