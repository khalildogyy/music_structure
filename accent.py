import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

filename = music_path()
y, sr = librosa.load(filename,sr=None)

o_envs = librosa.onset.onset_strength(y, sr=sr)
times = librosa.frames_to_time(np.arange(len(o_envs)), sr=sr)

