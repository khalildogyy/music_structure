# -- coding: utf-8 --
# Track beats using time series input
import os
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from music_path import music_path
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# filename ='/Users/admin/Desktop/MyDownfall.mp3'
filename = music_path()
s, sr = librosa.load(filename,sr=None,duration=10)
s *= 10000

# # 每秒采样数
# print('sample ratio:',sr)
# # 总采样数
# print('s num:',len(s))
# print('s:',s)

#低通滤波：使⽤截止频率为3kHz的50阶低通滤波器进⾏平滑处理
order = 50
fs = sr       # sample rate, Hz
cutoff = 3000  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 10.0         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
data = s

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()

#降采样：每7个点取1个，将采样频率由44.1kHz降⾄至6.3kHz减⼩计算量


#取平方：信号求平方使幅度包络变为能量包络


#平滑：最后使⽤3点滑动平均滤波器反复进行平滑，至平滑前后峰值点的个数不再变化








