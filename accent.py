import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from music_path import music_path

filename = music_path()
y, sr = librosa.load(filename,sr=None)

o_envs = librosa.onset.onset_strength(y, sr=sr)
times = librosa.frames_to_time(np.arange(len(o_envs)), sr=sr)

#瞬时频率
frequencies, D = librosa.ifgram(y, sr=sr)




#mu率压缩
xn
mu = 100
yn = np.log(1 + mu * xn) / np.log(1 + mu)

#插值
cn = [0] * (2*len(yn) - 1)  
for i in range(len(yn)):
	cn[2*i] = yn[i]

#低通滤波



#半波整流


