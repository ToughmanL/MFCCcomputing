
import numpy
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

sample_rate, signal = wavfile.read('8.wav')
signal = signal[0:int(3.5*sample_rate)]

axis_x=numpy.arange(0,signal.size,1)
plt.plot(axis_x,signal,linewidth=5)
plt.title("Time domain plot")
plt.xlabel("Time",fontsize = 14)
plt.ylabel("Amplitude",fontsize = 14)
plt.tick_params(axis='both',labelsize = 14)
plt.savefig('Time domain plot.png')
plt.show()


"""
Pre-Emphasis 预加重
第一步是对信号应用预加重滤波器，以放大高频。 预加重滤波器在几种方面有用：
（1）平衡频谱，因为高频通常比低频具有较小的幅度；
（2）避免在傅立叶变换操作期间出现数值问题；
（3）还可改善信号 噪声比（SNR）。
可以使用以下公式中的一阶滤波器将预加重滤波器应用于信号x：
            y(t)=x(t) -αx(t-1)
使用以下代码行即可轻松实现，其中滤波器系数（α）的典型值为0.95或0.97，
"""
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

axis_x=numpy.arange(0,emphasized_signal.size,1)
plt.plot(axis_x,emphasized_signal,linewidth=5)
plt.title("Pre-Emphasis")
plt.xlabel("Time",fontsize = 14)
plt.ylabel("Amplitude",fontsize = 14)
plt.tick_params(axis='both',labelsize = 14)
plt.savefig("Pre-Emphasis.png")
plt.show()


"""
经过预加重后，我们需要将信号分成短帧。 此步骤的基本原理是信号中的频率会随时间变化，
因此在大多数情况下，对整个信号进行傅立叶变换是没有意义的，
因为我们会随时间丢失信号的频率轮廓。 
为避免这种情况，我们可以假设信号的频率在很短的时间内是固定的。 
因此，通过在此短帧上进行傅立叶变换，可以通过串联相邻帧来获得信号频率轮廓较好的近似。
语音处理中的典型帧大小为20毫秒至40毫秒，连续帧之间有50％（+/- 10％）重叠。 
常见的设置是帧大小为25毫秒，frame_size = 0.025和10毫秒跨度（重叠15毫秒），
"""
frame_stride = 0.01
frame_size = 0.025
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z)
# 填充信号以确保所有帧具有相同数量的样本，而不会截断原始信号中的任何样本

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]

"""
将信号切成帧后，我们对每个帧应用诸如汉明窗之类的窗口函数。 Hamming窗口具有以下形式：
            w[n]=0.54-0.46cos(2*pi*n/(N-1))
其中0<=n<=N-1, N是窗长
有很多原因需要将窗函数应用于这些帧，特别是要抵消FFT无限计算并减少频谱泄漏
"""
frames *= numpy.hamming(frame_length)
#frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

#傅立叶变换和功率谱
"""
现在，我们可以在每个帧上执行N点FFT来计算频谱，这也称为短时傅立叶变换（STFT），
其中N通常为256或512，NFFT = 512； 然后使用以下公式计算功率谱（周期图）：
            P=|FFT(xi)|^2/N
其中，xi是信号x的第i帧。 这可以用以下几行实现：
"""
NFFT = 512
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

#滤波器组 Filter Banks
"""
计算滤波器组的最后一步是将三角滤波器（通常为40个滤波器，在Mel等级上为nfilt = 40）应用于功率谱以提取频带。 
梅尔音阶的目的是模仿低频的人耳对声音的感知，方法是在较低频率下更具判别力，而在较高频率下则具有较少判别力。
 我们可以使用以下公式在赫兹（f）和梅尔（m）之间转换：
            m = 2595log10(1+f/700)
            f = 700*(10^(m/2595)-1)
"""
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)  # dB

plt.title("filter_banks")
plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.1, extent=[0,filter_banks.shape[1],0,filter_banks.shape[0]]) #画热力图
plt.xlabel("Frames",fontsize = 14)
plt.ylabel("Dimension",fontsize = 14)
plt.tick_params(axis='both',labelsize = 14)
plt.savefig('filter_banks.png')
plt.show()


#梅尔倒谱Mel-frequency Cepstral Coefficients (MFCCs)
"""
事实证明，在上一步中计算出的滤波器组系数是高度相关的，这在某些机器学习算法中可能会出现问题。 
因此，我们可以应用离散余弦变换（DCT）去相关滤波器组系数，并产生滤波器组的压缩表示。 
通常，对于自动语音识别（ASR），结果倒谱系数2-13将保留，其余的将被丢弃； num_ceps =12。
丢弃其他系数的原因是它们代表滤波器组系数的快速变化，而这些细微的细节对自动语音识别（ASR）毫无帮助。
"""
num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

"""
可以将正弦提升器1应用于MFCC，去加重过高的MFCCs，这被可以改善嘈杂信号中的语音识别。
"""
cep_lifter=22
(nframes, ncoeff) = mfcc.shape
n = numpy.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
mfcc *= lift  #*

plt.title("mfcc")
plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.05, extent=[0,mfcc.shape[1],0,mfcc.shape[0]]) #画热力图
plt.xlabel("Frames",fontsize = 14)
plt.ylabel("Dimension",fontsize = 14)
plt.tick_params(axis='both',labelsize = 14)
plt.savefig('mfcc.png')
plt.show()


#平均归一化Mean Normalization
"""
如前所述，为了平衡频谱并改善信噪比（SNR），我们可以简单地从所有帧中减去每个系数的平均值。
"""
filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

plt.savefig("filter_banks_mean.png")
plt.title("filter_banks_mean")
plt.imshow(numpy.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.1, extent=[0,filter_banks.shape[1],0,filter_banks.shape[0]]) #画热力图
plt.xlabel("Frames",fontsize = 14)
plt.ylabel("Dimension",fontsize = 14)
plt.tick_params(axis='both',labelsize = 14)
plt.savefig('filter_banks_mean.png')
plt.show()

plt.title("mfcc_mean")
plt.imshow(numpy.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.05, extent=[0,mfcc.shape[1],0,mfcc.shape[0]]) #画热力图
plt.xlabel("Frames",fontsize = 14)
plt.ylabel("Dimension",fontsize = 14)
plt.tick_params(axis='both',labelsize = 14)
plt.savefig('mfcc_mean.png')
plt.show()