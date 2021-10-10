import pickle
import sys

MODEL = 0
THRESHOLD = float(sys.argv[1])

with open(f'results/{THRESHOLD}.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    same_correct,same_false,different_correct,different_false = pickle.load(f)

print(f'Total Correct: {same_correct + different_correct}, Total Incorrect: {same_false + different_false}')
print(f'False Negative: {same_false/(80*40)}')
print(f'False Positive: {different_false/(78*40)}')