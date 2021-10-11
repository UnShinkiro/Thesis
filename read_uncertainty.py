import pickle
import csv
import sys
import matplotlib.pyplot as plt
from operator import add

THRESHOLD = float(sys.argv[1])

with open(f'results/uncertainty_{THRESHOLD}.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    decisions = pickle.load(f)

with open('test_utterance.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    utterance, spk_list = pickle.load(f)

correct = 0
Incorrect = 0
failed_correct = 0
passed_correct = 0
correct_epistemic = []
correct_aleatoric = []
incorrect_epistemic = []
incorrect_aleatoric = []
false_negative_epistemic = []
false_negative_aleatoric = []

for claimed_speaker in spk_list:
    for speaker in spk_list:
        if speaker == claimed_speaker:
            count = 40
        else:
            count = 1
        for file in utterance[speaker]['files'][0:count]:
            if decisions[file]['decision'] == decisions[file]['actual']:
                correct += 1
                correct_epistemic.append(decisions[file]['epistemic'])
                correct_aleatoric.append(decisions[file]['aleatoric'])
                if decisions[file]['actual'] == True:
                    passed_correct += 1
            else:
                Incorrect += 1
                incorrect_epistemic.append(decisions[file]['epistemic'])
                incorrect_aleatoric.append(decisions[file]['aleatoric'])
                if decisions[file]['actual'] == True:
                    failed_correct += 1
                    false_negative_epistemic.append(decisions[file]['epistemic'])
                    false_negative_aleatoric.append(decisions[file]['aleatoric'])

print(passed_correct, failed_correct)

correct_total = list(map(add, correct_aleatoric, correct_epistemic))
incorrect_total = list(map(add, incorrect_aleatoric, incorrect_epistemic))
'''
plt.show()
plt.hist(incorrect_epistemic,bins=25)
plt.show()
plt.hist(incorrect_aleatoric,bins=25)
plt.show()
plt.hist(incorrect_total,bins=25)
plt.show()

plt.hist(correct_epistemic,bins=25)
plt.ylabel('Number of correct')
plt.xlabel('Amount of epistemic uncertainty')
plt.title('Correct Classification\'s epestemic uncertainty')
plt.show()

plt.hist(correct_aleatoric,bins=25)
plt.ylabel('Number of correct')
plt.xlabel('Amount of aleatoric uncertainty')
plt.title('Correct Classification\'s aleatoric uncertainty')
plt.show()

plt.hist(correct_total,bins=25)
plt.ylabel('Number of correct')
plt.xlabel('Amount of total uncertainty')
plt.title('Correct Classification\'s total uncertainty')
plt.show()

plt.hist(false_negative_epistemic,bins=20)
plt.show()

plt.hist(false_negative_aleatoric,bins=20)
plt.show()

false_negative_total = list(map(add, false_negative_aleatoric, false_negative_epistemic))
plt.hist(false_negative_total,bins=20)
plt.show()
'''
'''
with open('uncertainty_hist_for_correct.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(correct_epistemic)
    writer.writerow(correct_aleatoric)
'''
print(correct, Incorrect)
# d-vector for same speaker very less likely to be similar for text-indpedent, therefore high uncertainty