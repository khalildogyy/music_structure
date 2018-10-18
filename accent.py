import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

filename = music_path()
y, sr = librosa.load(filename,sr=None)

o_envs = librosa.onset.onset_strength(y, sr=sr)
times = librosa.frames_to_time(np.arange(len(o_envs)), sr=sr)

xn
mu = 0.1
yn = np.log(1 + mu * xn) / np.log(1 + mu)

#插值
zn = [0] * (2*len(yn) - 1)  
for i in range(len(yn)):
	zn[2*i] = yn[i]