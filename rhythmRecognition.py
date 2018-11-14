# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 15:38
# @Author  : UNE
# @Project : music_structure
# @File    : rhythmRecognition.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5, btype='lowpass'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5, btype='lowpass'):
    b, a = butter_lowpass(cutoff, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
    return y

# 分帧
def splitFrame(data, T, frameTime, overlap):
    framelength = int((len(data) / T) / (1000 / frameTime))
    overlength = int(framelength * overlap)

    frameindex = []
    for i in range(0, data.shape[0], overlength):
        if i + framelength > data.shape[0]:
            break
        frameindex.append([i, i + framelength])

    dataFrames = []
    for fi in frameindex:
        dataFrames.append(data[fi[0]: fi[1]])
    dataFrames = np.array(dataFrames, dtype=np.float32)
    return dataFrames

# 欧式距离相似度
def computeSimilarity(data):
    A = data
    B = np.transpose(data)
    similarMatrix = np.sqrt(np.abs(np.dot(np.square(A), np.ones_like(B)) + np.dot(np.ones_like(A), np.square(B)) - 2 * np.dot(A, B)))
    return similarMatrix

class RhythmRecognition():
    def __init__(self, filepath, duration=20):
        self.filepath = filepath
        self.data, self.sr = librosa.load(self.filepath, sr=None, duration=duration)
        self.T = duration

    """ 瞬时功率 """
    def instantaneousPower(self, order=6, cutoff=3000, downnum=7, need_plot=False):
        # 画出瞬时功率图
        def plot(fs, T, origindata, filterdata, downsampling, square):
            # frequency
            b, a = butter_lowpass(cutoff, fs, order)
            w, h = freqz(b, a, worN=8000)

            # Plot the frequency response.
            fig, axes = plt.subplots(4, 1, figsize=(10, 8))

            t1 = np.linspace(0, T, origindata.shape[0], endpoint=False)
            t2 = np.linspace(0, T, downsampling.shape[0], endpoint=False)

            axes[0].plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
            axes[0].plot(cutoff, 0.5 * np.sqrt(2), 'ko')
            axes[0].axvline(cutoff, color='k')
            axes[0].set_xlim(0, 0.5 * fs)
            axes[0].set_title("Lowpass Filter Frequency Response")
            axes[0].set_xlabel('Frequency [Hz]')
            axes[0].grid()
            axes[1].plot(t1, origindata, 'b-', label='data')
            axes[1].plot(t1, filterdata, 'g-', linewidth=0.5, label='filtered data')
            axes[1].set_xlabel('Time [sec]')
            axes[1].grid()
            axes[1].legend()
            axes[2].plot(t2, downsampling, 'g-', linewidth=1, label='downsampling data')
            axes[2].set_xlabel('Time [sec]')
            axes[2].grid()
            axes[2].legend()
            axes[3].plot(t2, square, 'g-', linewidth=1, label='square data')
            axes[3].set_xlabel('Time [sec]')
            axes[3].grid()
            axes[3].legend()

            plt.show()

        # 低通滤波：使⽤截止频率为3kHz的50阶低通滤波器进⾏平滑处理
        filterdata = butter_lowpass_filter(self.data, cutoff, self.sr, order)

        # 降采样：每7个点取1个，将采样频率由44.1kHz降⾄至6.3kHz减⼩计算量
        downsampling = filterdata[range(0, filterdata.shape[0], downnum)]

        # 取平方：信号求平方使幅度包络变为能量包络
        square = np.square(downsampling)

        if need_plot:
            plot(self.sr, self.T, self.data, filterdata, downsampling, square)

        return square

    """ NAE 信号 """
    def NAEsignal(self, need_plot=False):
        # 画出瞬时功率图
        def plot(times, nae, onset_frames, T):
            fig, ax = plt.subplots(figsize=(6, 5))
            t1 = np.linspace(0, T, nae.shape[0], endpoint=False)
            ax.plot(t1, nae, label='NAE')
            ax.vlines(times[onset_frames], 0, NAE.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
            ax.set_xlabel('Time [sec]')
            ax.axis('tight')
            ax.legend(frameon=True, framealpha=0.75)
            plt.show()

        # 瞬时功率
        square = self.instantaneousPower()

        # 获取开始强度点
        o_envs = librosa.onset.onset_strength(self.data, sr=self.sr)
        times = librosa.frames_to_time(np.arange(o_envs.shape[0]), sr=self.sr)

        # 按键强度
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_envs, sr=self.sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)

        frames_times = np.append(onset_times, [0])
        frames_times = np.append(frames_times, times[-1])
        frames_times.sort()

        # 平均能量 NAE
        NAE = []
        for i in range(1, len(frames_times)):
            start = int(frames_times[i-1] * square.shape[0] / self.T)
            end = int(frames_times[i] * square.shape[0] / self.T)
            temp = square[start:end]  # 存onset_strength
            tempnae = []
            for j in range(len(temp)):
                tempnae.append(np.sum(temp[j:]))
            NAE += tempnae

        NAE.append(square[-1])
        NAE = np.array(NAE)

        if need_plot:
            plot(times, NAE, onset_frames, self.T)

        return NAE

    """ 重音信号 """
    def streeSignal(self, frameTime=23, overlap=0.5, need_plot=False):
        # 重音信号图
        def plot(T, originframes, dft_frames, smooth_frames, weight_frames, stresssignal):
            fig, axes = plt.subplots(5, 1, figsize=(10, 11))

            t = np.linspace(0, T, originframes.shape[0], endpoint=False)

            axes[0].plot(t, originframes)
            axes[0].set_title("Origin frames")
            axes[0].set_xlabel('Time [sec]')
            axes[1].plot(t, dft_frames)
            axes[1].set_title("Discrete Fourier Transform")
            axes[1].set_xlabel('Time [sec]')
            axes[2].plot(t, smooth_frames)
            axes[2].set_title("Smooth Signal")
            axes[2].set_xlabel('Time [sec]')
            axes[3].plot(t, weight_frames)
            axes[3].set_title("Weigth Sum")
            axes[3].set_xlabel('Time [sec]')
            axes[4].plot(t, stresssignal)
            axes[4].set_title("Stress Signal")
            axes[4].set_xlabel('Time [sec]')

            plt.show()

        # 获取帧
        frames = splitFrame(self.data, self.T, frameTime, overlap)

        # 每一帧离散傅里叶变换
        frames_dft = np.fft.fft(frames, axis=1)

        # μ率压缩
        mu = 100
        frames_mu = np.log(1 + mu * frames_dft) / np.log(1 + mu)

        # 截止频率为10Hz的6阶巴特沃兹滤波器 平滑
        smooth_frames = butter_lowpass_filter(frames_mu, cutoff=10000, fs=self.sr, order=8)

        # 差分
        frames_diff = np.diff(smooth_frames, axis=1)
        # 半波整流
        frames_halfwave = np.copy(frames_diff)
        frames_halfwave[frames_halfwave < 0] = 0
        frames_halfwave = np.concatenate([np.zeros([frames_halfwave.shape[0], 1]), frames_halfwave], axis=1)

        # 获得加权信号
        lamda = 0.8
        frames_weight = frames_halfwave + lamda * smooth_frames

        # 获得重音信号
        stessSignal = np.max(frames_weight, axis=1)

        if need_plot:
            plot(self.T, frames, frames_dft, smooth_frames, frames_weight, stessSignal)


        return stessSignal

    """ 计算拍谱 """
    def rhythmSpectum(self, data, frameTime=40, overlap=0.5, need_plot=False, plotname=''):
        # 画出拍谱图
        def plot(beats, T):
            t = np.linspace(0, T, len(beats), endpoint=False)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(t, beats, label='Beat curve')
            ax.axis('tight')
            ax.set_xlabel('Time [sec]')
            ax.set_title(plotname)
            plt.show()

        frames = splitFrame(data, self.T, frameTime, overlap)

        # 计算帧间相似性
        matrix = computeSimilarity(frames)
        print("分帧之后： {}, 相似矩阵 {}".format(frames.shape, matrix.shape))

        # 计算拍谱
        beats = np.cumsum(matrix, axis=1)

        # plot
        if need_plot:
            plot(beats, self.T)

        return beats

if __name__ == '__main__':
    testrr = RhythmRecognition(filepath='./data/MyDownfall.mp3')

    instant = testrr.instantaneousPower(need_plot=True)
    nae = testrr.NAEsignal(need_plot=True)
    stress = testrr.streeSignal(need_plot=True)
    print("瞬时功率 {}，NAE {}， 重音 {}".format(instant.shape, nae.shape, stress.shape))

    instant_beats = testrr.rhythmSpectum(instant, need_plot=True, plotname='Instant power')
    nae_beats = testrr.rhythmSpectum(nae, need_plot=True, plotname='NAE')
    stess_beats = testrr.rhythmSpectum(stress, need_plot=True, plotname='Stess Signal')

