from scipy.io import wavfile
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

INT16_MAX = 32767
SAMPLE_RATE,_ = wavfile.read("../VCTK-Corpus/wav48/p225/p225_001.wav")
FRAME_SIZE = int(SAMPLE_RATE*0.01)              # 10ms
INPUT_FRAME_SIZE = 3*FRAME_SIZE                 # left + self + right frame      

'''
sample_rate, data = wavfile.read("../VCTK-Corpus/wav48/p225/p225_001.wav")
#print(sample_rate)
frame_size = int(sample_rate/100)           # 10ms frame
nframes = int(data.size/frame_size) + 1
#print(nframes)

for n in range(nframes):
    frame = data[n*frame_size : (n+1)*frame_size]
    if frame.size < frame_size:
        np.zeros(frame_size - frame.size, dtype=int)
        frame = np.concatenate((frame,np.zeros(frame_size - frame.size, dtype=int)))
'''

spk_list = os.listdir("../VCTK-Corpus/wav48/")
#print(pid)
utterance = {}
training_data = []
speaker = []
for pid, speaker in enumerate(spk_list[0:10]):
    utterance[speaker] = {}
    path = "../VCTK-Corpus/wav48/" + speaker
    utterance[speaker]['files'] = os.listdir(path)
    for count, files in enumerate(utterance[speaker]['files']):
        if count < 30:
                file_path = "../VCTK-Corpus/wav48/" + speaker + "/" + files
                _, data = wavfile.read(file_path)         # requires tons of memory
                data = data/INT16_MAX
                training_data.append((data,pid))

for entry in training_data[0:2]:
    data, label = entry
    nframes = int(data.size/FRAME_SIZE) + 1
    for n in range(1,nframes):
        frame = data[(n-1)*FRAME_SIZE : (n+2)*FRAME_SIZE]
        print(frame.size)
        if frame.size < FRAME_SIZE*3:
            np.zeros(FRAME_SIZE*3 - frame.size, dtype=int)
            frame = np.concatenate((frame,np.zeros(FRAME_SIZE*3 - frame.size, dtype=int)))
            print(frame)



#print(training_data[0:2])
'''
print(type(utterance["p229"]["p229_308.wav"]))
print(utterance["p229"]["p229_308.wav"].dtype)      #int16 max = 32767
print(max(utterance["p229"]["p229_308.wav"]))
print(min(utterance["p229"]["p229_308.wav"]))
'''
'''
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(3*frame_size,)),  # input layer
    keras.layers.Dense(256, activation='relu'),  # hidden layer (1)
    keras.layers.Dense(256, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(256, activation='relu'),  # hidden layer (3)
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),  # hidden layer (4)
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax') # output layer
])
'''