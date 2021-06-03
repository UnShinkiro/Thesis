from scipy.io import wavfile
import os
import numpy as np

sample_rate, data = wavfile.read("../VCTK-Corpus/wav48/p225/p225_001.wav")
print(sample_rate)
frame_size = int(sample_rate/100)           # 10ms frame
nframes = int(data.size/frame_size) + 1
print(nframes)

for n in range(nframes):
    frame = data[n*frame_size : (n+1)*frame_size]
    if frame.size < frame_size:
        np.zeros(frame_size - frame.size, dtype=int)
        frame = np.concatenate((frame,np.zeros(frame_size - frame.size, dtype=int)))

pid = os.listdir("../VCTK-Corpus/wav48/")
print(pid)
utterance = {}
for speaker in pid[0:10]:
    utterance[speaker] = {}
    path = "../VCTK-Corpus/wav48/" + speaker
    utterance[speaker]['files'] = os.listdir(path)
    for files in utterance[speaker]['files']:
        file_path = "../VCTK-Corpus/wav48/" + speaker + "/" + files
        _, utterance[speaker][files] = wavfile.read(file_path)         # requires tons of memory

print(utterance["p229"]["p229_308.wav"])