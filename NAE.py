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
o_envs = librosa.onset.onset_strength(y, sr=sr)
times = librosa.frames_to_time(np.arange(len(o_envs)), sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_envs, sr=sr)
Frames_times = librosa.frames_to_time(onset_frames, sr=sr)
print(onset_frames)
print(Frames_times)

# dict(map(lambda x,y:[x,y], times, o_envs))

frames_times = np.append(Frames_times,[0])
frames_times = np.append(frames_times,times[-1])
frames_times.sort()
print(frames_times)

# Onset_Frames = np.append(onset_frames,[0])
# Onset_Frames.sort()


#平均能量 NAE
NAE = []
# i = 0
# k = 0
# energy = 0

# for i, time in enumerate(times):
# 	print('frames_times:',frames_times[i])
# 	if time != frames_times[i]:
# 		j = times.tolist().index(time)
# 		print('j:',j)
# 		print('time:',time)
# 		energy += o_envs[j]
# 		print('energy:',energy)
# 	elif time == frames_times[i]:

# 		break

for i in range(1, len(frames_times)):
	start = np.where(times == frames_times[i-1])[0][0] #取下标
	end = np.where(times == frames_times[i])[0][0]
	temp = o_envs[start:end] #存onset_strength
	tempnae = []
	for j in range(len(temp)):
		tempnae.append(np.sum(temp[j:]))
	NAE += tempnae

NAE.append(o_envs[-1])
NAE = np.array(NAE)

# print(NAE)


# D = np.abs(librosa.stft(y))
plt.figure()
plt.subplot(1, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),x_axis='time', y_axis='log')
plt.plot(times, NAE, label='Onset strength')
plt.vlines(times[onset_frames], 0, NAE.max(), color='r', alpha=0.9,linestyle='--', label='Onsets')
plt.axis('tight')
plt.legend(frameon=True, framealpha=0.75)
plt.show()



