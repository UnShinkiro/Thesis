import os
import pickle
import numpy as np

with open('test_utterance.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    utterance, spk_list = pickle.load(f)

pre_emphasis = 0.97
claimed_speaker = spk_list[0]
actual_speaker = spk_list[5]

test_file_path = 'vox/vox1_test_wav/' + actual_speaker + '/' + utterance[actual_speaker]['files'][0]
intermediate_layer_model = []
intermediate_output = []

_, data = wavfile.read(test_file_path)
emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
input_data = []
input_label = []   # Not used
form_input_data((emphasized_signal,0), input_data, input_label)

for n in range(5):
    print(f'Loading DNN model {n}')
    model = tf.keras.models.load_model(f"saved_model/{n}")
    model.summary()
    layer_name = 'dropout_1'
    intermediate_layer_model.append(keras.models.Model(inputs=model.input,
                                                    outputs=model.get_layer(layer_name).output))

    print(f'extracting d-vector from test file using model {n}')
    intermediate_output[n] = intermediate_layer_model[n].predict(np.array(input_data))
    d_test = np.zeros(256)
    for out in intermediate_output[n]:
        d_test += out/sum(out)

    print(f'loading speaker d-vector for model {n}')
    filename = f'd-vector/{n}' +  claimed_speaker + '.pkl'
    with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
        d_utterance_list, d_model = pickle.load(f)
        d_vectors[n]['d_utterance_list'] = d_utterance_list
        d_vectors[n]['d_model'] = d_model
        d_vectors[n]['d_test'] = d_test
    
# Extract Uncerntainties


