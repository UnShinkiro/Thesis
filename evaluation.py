from scipy.io import wavfile
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pre_process import form_input_data
pre_emphasis = 0.97
correct_count = 0
incorrect_count = 0

model = tf.keras.models.load_model("saved_model/my_model")
model.summary()
layer_name = 'dropout_1'
intermediate_layer_model = keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

with open('test_utterance.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    utterance, spk_list = pickle.load(f)

for chosen_speaker in spk_list:
    print(f"testing speaker {chosen_speaker}")
    filename = 'd-vector/' +  chosen_speaker + '.pkl'
    with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
        d_utterance_list, d_model = pickle.load(f)

    for speaker in spk_list:
        print(f"test with {speaker}'s wavfiles")
        for file in utterance[speaker]['files']:
            _, data = wavfile.read("vox/vox1_test_wav/" + speaker + "/" + file)
            emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
            evaluation_data = []
            evaluation_label = []
            form_input_data((emphasized_signal,pid), evaluation_data, evaluation_label)
            intermediate_output = intermediate_layer_model.predict(np.array(evaluation_data))
            d_eva = np.zeros(256)
            for out in intermediate_output:
                d_eva += out/sum(out)
            if np.corrcoef(d_model,d_eva)[0][1] >= 0.8:
                # Same speaker
                if speaker == chosen_speaker:
                    correct_count += 1
                else:
                    incorrect_count += 1
            else:
                # Different speaker
                if not speaker == chosen_speaker:
                    correct_count += 1
                else:
                    incorrect_count += 1

print(correct_count, incorrect_count)