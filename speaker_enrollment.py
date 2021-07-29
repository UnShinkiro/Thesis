from scipy.io import wavfile
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pre_process import form_input_data
pre_emphasis = 0.97

model = tf.keras.models.load_model("saved_model/my_model")
model.summary()
layer_name = 'dropout_1'
intermediate_layer_model = keras.models.Model(  inputs=model.input,
                                                outputs=model.get_layer(layer_name).output)

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
    enrollment_dataset = []
    for count in range(5):
        file_path = "vox/vox1_test_wav/" + speaker + "/" + utterance[speaker]['files'].pop(0)
        _, data = wavfile.read(file_path)         # requires tons of memory with many spekaers
        emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
        enrollment_dataset.append((emphasized_signal,pid))

    enrollment_data = []
    enrollment_label = [] #Not used
    d_utterance_list = []
    
    for entry in enrollment_dataset:
        enrollment_data.clear()
        enrollment_label.clear()
        form_input_data(entry, enrollment_data, enrollment_label)
        intermediate_output = intermediate_layer_model.predict(np.array(enrollment_data))
        d_utterance = np.zeros(256)
        for out in intermediate_output:
            d_utterance += out/sum(out)
        d_utterance_list.append(d_utterance) # Saving the utterance d-vector for future uncertainty measure
    d_model = np.zeros(256)
    for vector in d_utterance_list:
        d_model += vector
    d_model = d_model/len(d_utterance_list)
    filename = 'd-vector/' + speaker + '.pkl'
    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([d_utterance_list, d_model], f)

with open('test_utterance,pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([utterance, spk_list], f)

