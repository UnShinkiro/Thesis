import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.io import wavfile
from pre_process import form_input_data
from sklearn.neighbors import KernelDensity
import matplotlib
import matplotlib.pyplot as plt

with open('test_utterance.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    utterance, spk_list = pickle.load(f)

pre_emphasis = 0.97
N_MODELS = 20
THRESHOLD = float(sys.argv[1])
decision_and_uncertainty = {}

intermediate_layer_model = []

# Load models
for n in range(N_MODELS):
    print(f'Loading DNN model {n}')
    model = tf.keras.models.load_model(f"saved_model/{n}")
    #model.summary()
    layer_name = 'dropout_1'
    intermediate_layer_model.append(keras.models.Model(inputs=model.input,
                                                    outputs=model.get_layer(layer_name).output))

for claimed_speaker in spk_list:
    for speaker in spk_list:
        print(f'using speaker {speaker}\'s file')
        if speaker == claimed_speaker:
            count = 40
        else:
            count = 1
        for file in utterance[speaker]['files'][0:count]:
            print(f'evaluation using {file}')
            # Extract d_test (d-vector for test utterance)
            intermediate_output = []
            d_vectors = {}
            _, data = wavfile.read('vox/vox1_test_wav/' + speaker + '/' + file)
            emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
            input_data = []
            input_label = []   # Not used
            form_input_data((emphasized_signal,0), input_data, input_label)

            for n in range(N_MODELS):
                #print(f'extracting d-vector from test file using model {n}')
                intermediate_output.append(intermediate_layer_model[n].predict(np.array(input_data)))
                d_test = np.zeros(256)
                for out in intermediate_output[n]:
                    d_test += out/sum(out)

                #print(f'loading speaker d-vector for model {n}')
                filename = f'd-vector/{n}/' +  claimed_speaker + '.pkl'
                with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
                    d_utterance_list, d_model = pickle.load(f)
                    d_vectors[n] = {}
                    d_vectors[n]['d_utterance_list'] = d_utterance_list
                    d_vectors[n]['d_model'] = d_model
                    d_vectors[n]['d_test'] = d_test

            # Extract Uncerntainties
            #print('extracting uncertainties')
            scores = []
            for n in range(N_MODELS):
                scores.append(np.corrcoef(d_vectors[n]['d_model'],d_vectors[n]['d_test'])[0][1])
            mean_score = np.mean(scores)
            # adjusting degree of support weighting function threshold according to speaker model score

            if(np.mean(scores) > THRESHOLD):
                plausibility = scores/max(scores)
            else:
                scores = np.ones(len(scores))-np.absolute(scores)
                plausibility = (scores)/max(scores)

            # Kernel Density Estimation for Degree of support
            probability = np.zeros(N_MODELS)
            for n in range(N_MODELS):
                score = []
                weight = []
                for d_utterance in d_vectors[n]['d_utterance_list']:
                    corr = np.corrcoef(d_utterance,d_vectors[n]['d_test'])[0][1]
                    score.append(corr)
                    if mean_score > THRESHOLD:
                        if corr > THRESHOLD:
                            weight.append((corr-THRESHOLD)/(1-THRESHOLD) * 10)
                        else:
                            weight.append((THRESHOLD - abs(corr))/THRESHOLD)
                    else:
                        if corr > THRESHOLD:
                            weight.append((corr - THRESHOLD)/(1-THRESHOLD) * 10)
                        else:
                            weight.append(abs(mean_score - THRESHOLD - abs(corr)))
                #weight = np.absolute(mean_score-np.absolute(score))
                score = np.reshape(score, (-1,1))
                std = np.std(score)
                kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(score,y=None,sample_weight = weight)
                X_axis = np.linspace(-1,2,1000)
                density = np.exp(kde.score_samples(np.reshape(X_axis,(-1,1))))
                density = density/sum(density)
                #plt.plot(X_axis,density)
                #plt.show()
                #print(score)
                #print(weight)
                for k in range(len(density)):
                    if X_axis[k] > THRESHOLD:
                        probability[n] += density[k]
                #print(probability[n])

            DOS_pos = np.zeros(N_MODELS)
            DOS_neg = np.zeros(N_MODELS)
            for n in range(N_MODELS):
                DOS_pos[n] = max(2*probability[n] - 1,0)
                DOS_neg[n] = max(1 - 2*probability[n],0)
            Plau_pos = np.zeros(N_MODELS)
            Plau_neg = np.zeros(N_MODELS)
            for n in range(N_MODELS):
                Plau_pos[n] = min(plausibility[n],DOS_pos[n])
                Plau_neg[n] = min(plausibility[n],DOS_neg[n])
            P_pos = max(Plau_pos)
            P_neg = max(Plau_neg)
            epistemic = min(P_pos,P_neg)
            aleatoric = 1 - max(P_pos,P_neg)
            decision_and_uncertainty[file] = {}
            decision = True
            if P_pos >= P_neg:
                decision = True
            else:
                decision = False
            actual = True
            if speaker == claimed_speaker:
                actual = True
            else:
                actual = False
            decision_and_uncertainty[file]['decision'] = decision
            decision_and_uncertainty[file]['epistemic'] = epistemic
            decision_and_uncertainty[file]['aleatoric'] = aleatoric
            decision_and_uncertainty[file]['actual'] = actual

with open(f'results/uncertainty_{THRESHOLD}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(decision_and_uncertainty, f)