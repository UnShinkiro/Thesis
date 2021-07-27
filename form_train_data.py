#from pre_process import form_input_data
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

for pid, speaker in enumerate(spk_list[0:100]):
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
            print("Add entry")
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

print(train_data[0:10])

with open('trainning_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([train_data, train_label, validation_data, validation_label, N_SPEAKER], f)
with open('utterance_list.pkl', 'wb') as f:
    pickle.dump([utterance, spk_list], f)

def form_input_data(entry, data_list, label_list):
    # 40 filter_banks + 30 frames left + 10 frames right
    data, spk = entry
    filter_banks = get_filter_banks(data)
    for n in range(30, len(filter_banks) - 10):
        frame = filter_banks[n-30: n+11].reshape(41*40)
        data_list.append(frame)
        label_list.append(spk)

def get_filter_banks(data):
    all_filter_banks = []
    nframes = int(data.size/FRAME_SIZE) + 1
    for n in range(nframes):
        frame = data[n*FRAME_SIZE : (n+1)*FRAME_SIZE]
        if frame.size < FRAME_SIZE:
            frame = np.concatenate((frame,np.zeros(FRAME_SIZE - frame.size, dtype=int)))
        all_filter_banks.append(extract_filter_banks(frame))
    return np.array(all_filter_banks)

def extract_filter_banks(frame):
    frame = frame * np.hamming(len(frame))
    # STFT
    mag_frame = np.absolute(np.fft.rfft(frame, NFFT))  # Magnitude of the FFT
    pow_frame = ((1.0 / NFFT) * ((mag_frame) ** 2))  # Power Spectrum
    # Filter bank
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, NFILT + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bank_bin = np.floor((NFFT + 1) * hz_points / SAMPLE_RATE)

    fbank = np.zeros((NFILT, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, NFILT + 1):
        f_m_minus = int(bank_bin[m - 1])   # left
        f_m = int(bank_bin[m])             # center
        f_m_plus = int(bank_bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bank_bin[m - 1]) / (bank_bin[m] - bank_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bank_bin[m + 1] - k) / (bank_bin[m + 1] - bank_bin[m])
    filter_banks = np.dot(pow_frame, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    return filter_banks