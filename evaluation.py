from scipy.io import wavfile
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pre_process import form_input_data

MODEL = 0
THRESHOLD = float(sys.argv[1])
print(THRESHOLD)
pre_emphasis = 0.97
same_correct = 0
same_false = 0
different_correct = 0
different_false = 0

model = tf.keras.models.load_model(f"saved_model/{MODEL}")
#model.summary()
layer_name = 'dropout_1'
intermediate_layer_model = keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

with open('test_utterance.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    utterance, spk_list = pickle.load(f)

for chosen_speaker in spk_list:
    print(f"testing speaker {chosen_speaker}")
    filename = f'd-vector/{MODEL}/' +  chosen_speaker + '.pkl'
    with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
        d_utterance_list, d_model = pickle.load(f)

    for speaker in spk_list:
        if speaker == chosen_speaker:
            print(f"\ttest with same speaker's wavfiles")    
            for file in utterance[speaker]['files'][0:80]:
                print(f"\t\ttesting file {file}")
                _, data = wavfile.read("vox/vox1_test_wav/" + speaker + "/" + file)
                emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
                evaluation_data = []
                evaluation_label = []   # Not used
                form_input_data((emphasized_signal,0), evaluation_data, evaluation_label)
                intermediate_output = intermediate_layer_model.predict(np.array(evaluation_data))
                d_eva = np.zeros(256)
                for out in intermediate_output:
                    d_eva += out/sum(out)
                if np.corrcoef(d_model,d_eva)[0][1] >= THRESHOLD:
                    same_correct += 1
                else:
                    same_false += 1
        else:
            print(f"\ttest with different speaker's wavfiles")
            for file in utterance[speaker]['files'][0:2]:
                print(f"\t\ttesting file {file}")
                _, data = wavfile.read("vox/vox1_test_wav/" + speaker + "/" + file)
                emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
                evaluation_data = []
                evaluation_label = []   # Not used
                form_input_data((emphasized_signal,0), evaluation_data, evaluation_label)
                intermediate_output = intermediate_layer_model.predict(np.array(evaluation_data))
                d_eva = np.zeros(256)
                for out in intermediate_output:
                    d_eva += out/sum(out)
                if np.corrcoef(d_model,d_eva)[0][1] >= THRESHOLD:
                    different_false += 1
                else:
                    different_correct += 1

with open(f'results/{THRESHOLD}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([same_correct,same_false,different_correct,different_false], f)