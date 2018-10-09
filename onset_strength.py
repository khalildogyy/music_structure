# -- coding: utf-8 --
# Track beats using time series input
import os
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

filename ='/Users/admin/Desktop/MyDownfall.mp3'
y, sr = librosa.load(filename,sr=None)

# #瞬时频率
# frequencies, D = librosa.ifgram(y, sr=sr)

# # 每秒采样数
# print('sample ratio:',sr)
# # 总采样数
# print('y num:',len(y))

# # get onset envelope
# onset_env = librosa.onset.onset_strength(y=y, sr=sr)
# print('onset_env length:',len(onset_env))
# print(onset_env)

#瞬时功率 onset_strength
o_env = librosa.onset.onset_strength(y, sr=sr)

times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

frames_times = np.append(onset_times,[0])
frames_times.sort()


# D = np.abs(librosa.stft(y))
plt.figure()
plt.subplot(1, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),x_axis='time', y_axis='log')
plt.plot(times, o_env, label='Onset strength')
plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,linestyle='--', label='Onsets')
plt.axis('tight')
plt.legend(frameon=True, framealpha=0.75)
plt.show()






