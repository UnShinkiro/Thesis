import pickle
import sys

MODEL = int(sys.argv[1])

with open(f'results/{MODEL}.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    correct_count, incorrect_count = pickle.load(f)

print(f'Correct: {correct_count}, Incorrect: {incorrect_count}')
print(f'Accuracy: {correct_count/(correct_count + incorrect_count)}')