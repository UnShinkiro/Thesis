# This filter_bank extraction function was taken from a online tutorial for Speech processing for
# Machine Learning. Citation as below.
'''
@misc{fayek2016,
  title   = "Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between",
  author  = "Haytham M. Fayek",
  year    = "2016",
  url     = "https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html"
}
'''

from scipy.io import wavfile
import os
import numpy as np
import matplotlib.pyplot as plt

NFFT = 512
NFILT = 40
FILE_PATH = "../VCTK-Corpus/wav48/p225/p225_002.wav"    # wav file path
SAMPLE_RATE, _ = wavfile.read(FILE_PATH)
FRAME_SIZE = int(SAMPLE_RATE * 0.025)    # use 25ms frame
pre_emphasis = 0.97

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

if __name__ == "__main__":
    _, data = wavfile.read(FILE_PATH)   # reads wav file
    t_axis = np.linspace(0,len(data)/SAMPLE_RATE,len(data))
    plt.plot(t_axis,data)
    plt.ylabel('Amplitude')
    plt.xlabel('time (sec)')
    emphasized_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1]) 
    # pre-emphasis to amplify high frequnecy signals
    plt.plot(t_axis,emphasized_signal)
    plt.ylabel('Amplitude')
    plt.xlabel('time (sec)')
    plt.legend(['original audio signal','pre-emphasised signal'])
    plt.title('Original signal vs pre-emphasised signal')
    plt.show()
    banks = []
    frequency = np.linspace(0,4,num=40)
    t = []
    nframes = int(data.size/FRAME_SIZE) + 1
    for n in range(30, nframes - 10):       # This frame setting is adopted from the d-vector paper
        frame = data[(n-30)*FRAME_SIZE: (n+11)*FRAME_SIZE]
        if frame.size < FRAME_SIZE*41:  # 41 because 30 left + 10 right + current frame
            frame = np.concatenate((frame,np.zeros(FRAME_SIZE*41 - frame.size, dtype=int)))
        filter_bank = extract_filter_banks(frame)
        banks.append(filter_bank)
        t.append(n*0.025)
    fig, ax0 = plt.subplots(nrows=1)
    cmap = plt.get_cmap('bwr')
    im = plt.pcolormesh(t,frequency,np.transpose(banks),shading='auto',cmap=cmap)  
    plt.ylabel('frequency (kHz)')
    plt.xlabel('time (sec)')
    plt.title('Filter banks of wav file')
    fig.colorbar(im,ax=ax0)
    plt.show()