from sklearn.neighbors import KernelDensity
import pickle
import numpy as np
import sys

N_MODELS = 20
THRESHOLD = float(sys.argv[1])
N_UTTERANCE = sys.argv[2]
decision_and_uncertainty = {}

with open('test_utterance.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    utterance, spk_list = pickle.load(f)

with open(f'results/{N_UTTERANCE}/d-vectors_{N_UTTERANCE}.pkl','rb') as f:
    d_vectors = pickle.load(f)

for claimed_speaker in spk_list:
    for speaker in spk_list:
        print(f'using speaker {speaker}\'s file')
        if speaker == claimed_speaker:
            count = 40
        else:
            count = 1
        for file in utterance[speaker]['files'][0:count]:
            # Extract Uncerntainties
            #print('extracting uncertainties')
            scores = []
            for n in range(N_MODELS):
                scores.append(np.corrcoef(d_vectors[claimed_speaker][n]['d_model'], \
                    d_vectors[speaker][file][n])[0][1])
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
                for d_utterance in d_vectors[claimed_speaker][n]['d_utterance_list']:
                    corr = np.corrcoef(d_utterance, d_vectors[speaker][file][n])[0][1]
                    score.append(corr)
                    if mean_score > THRESHOLD:
                        if corr > THRESHOLD:
                            weight.append((corr-THRESHOLD)/(1-THRESHOLD) * 10)
                        else:
                            weight.append(abs((THRESHOLD - abs(corr))/THRESHOLD))
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

with open(f'results/{N_UTTERANCE}/uncertainty_{THRESHOLD}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(decision_and_uncertainty, f)