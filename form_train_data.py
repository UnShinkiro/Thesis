from pre_process import form_input_data
from scipy.io import wavfile
import numpy as np
import os
import pickle

SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * 0.025)
NFFT = 512 
NFILT = 40
N_SPEAKER = 0
pre_emphasis = 0.97

spk_list = os.listdir("vox/vox1_dev_wav")
n = 0
while n in range(len(spk_list)):
    if spk_list[n].startswith("."):
        spk_list.pop(n)
        n -= 1
    n += 1

utterance = {}
emphasized_data = []
validation_dataset = []
validation_data = []
validation_label = []
train_data = []
train_label = []
enrollment_dataset = []
verification_dataset = []

for pid, speaker in enumerate(spk_list):
    N_SPEAKER += 1
    print(f"Logging speaker {pid}")
    utterance[speaker] = {}
    path = "vox/vox1_dev_wav/" + speaker
    folders = os.listdir(path)
    utterance[speaker]['files'] = []
    for folder in folders:
        if not folder.startswith("."):
            path = "vox/vox1_dev_wav/" + speaker + "/" + folder
            files = os.listdir(path)
            for file in files:
                if not file.startswith("."):
                    utterance[speaker]['files'].append(folder + "/" + file)

    for count in range(10):
        file_path = "vox/vox1_dev_wav/" + speaker + "/" + utterance[speaker]['files'].pop(0)
        _, data = wavfile.read(file_path)         # requires tons of memory with many spekaers
        emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
        if count < 5:
            emphasized_data.append((emphasized_signal,pid))
        elif count < 10:
            validation_dataset.append((emphasized_signal,pid))


counter = 0
for entry in emphasized_data:
    print(f"Handling entry {counter}")
    form_input_data(entry, train_data, train_label)
    counter += 1
for entry in validation_dataset:
    print(f"Handling entry {counter}")
    form_input_data(entry, validation_data, validation_label)
    counter += 1

with open('trainning_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([train_data, train_label], f)

with open('validation_data.pkl', 'wb') as f:
    pickle.dump([validation_data, validation_label], f)

with open('utterance_list.pkl', 'wb') as f:
    pickle.dump([utterance, spk_list, N_SPEAKER], f)
