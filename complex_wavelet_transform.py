"""
https://qiita.com/taront/items/96da49b1b1f512be5f98
"""


import numpy as np
import matplotlib as plt
import pywt
from typing import List, Tuple
import math


def get_ticks_label_set(all_labels: np.array, num: int):
    length = len(all_labels)
    step = length // (num - 1)

    pick_positions = np.arange(0, length, step)
    pick_labels = all_labels[::step]

    return pick_positions, pick_labels


def get_nyquist_freq(frequency: float = 1000) -> float:
    return frequency / 2.0


def sample_rate_to_freq(sample_rate=0.001) -> float:
    return 1 / sample_rate


def freq_to_sample_rate(frequency=1000) -> float:
    return 1 / frequency


def array_analyze_freq_liner(f_nq=get_nyquist_freq(), resolution=50):
    return np.linspace(1, f_nq, resolution)


def array_analyze_freq_log(f_nq=get_nyquist_freq(), resolution=50):
    return np.logspace(1, f_nq, resolution)


def get_times_analyzing(freq: float = 1000) -> np.array:
    dt = 1 / freq
    time = np.arange(-1, 1, dt)
    return time


def generate_signal(freq: float = 1000) -> np.array:
    t = get_nyquist_freq(freq)

    sig = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7 * (t - 0.4) ** 2) * np.exp(1j * 2 * np.pi * 2 * (t - 0.4)))

    return sig


def print_wavelet_forms():
    print(pywt.wavelist(kind='continuous'))
    return


def get_cmor_wavelet_type(bandwidth=1.5, center_frequency=1.0) -> str:
    return 'cmor' + str(bandwidth) + '-' + str(center_frequency)


def get_cmor_wavelet(bandwidth=1.5, center_freq=1.0):
    wavelet = pywt.ContinuousWavelet(get_cmor_wavelet_type(bandwidth, center_freq))
    return wavelet


def cmor_wavelet(bandwidth=1.5, center_freq=1.0, precision=8) -> Tuple[List, List]:
    int_psi, x = pywt.integrate_wavelet(get_cmor_wavelet(bandwidth, center_freq),
                                        precision=precision)

    return int_psi, x


def get_scales(sampling_rate=0.001):
    # sampling frequency and Nyquist frequency
    f_sample = sample_rate_to_freq(sampling_rate)
    f_nq = get_nyquist_freq(f_sample)

    freqs_analyze = array_analyze_freq_liner(f_nq, 50)
    freqs_ratio = freqs_analyze / f_sample
    scales = 1 / freqs_ratio
    scales = scales[::-1]

    return scales


def scale_conversion(sampling_rate=0.001, wavelet_type=get_cmor_wavelet_type(1.5, 1.0)) -> List:
    scales = get_scales(sampling_rate)

    frequencies_rate = pywt.scale2frequency(scale=scales, wavelet=wavelet_type)
    frequencies = frequencies_rate / sampling_rate

    return frequencies


def cwt(signal=generate_signal(), scales=get_scales(), wavelet_type=get_cmor_wavelet_type(1.5, 1.0))\
        -> Tuple[List, List]:
    cwtmat, freqs_rate = pywt.cwt(signal, scales=scales, wavelet=wavelet_type)
    return cwtmat, freqs_rate


def get_xpos_xlabel(freq=1000, num=10):
    t = get_times_analyzing(freq)
    x_pos, x_label = get_ticks_label_set(t, num)
    return x_pos, x_label


def get_ypos_ylabel(freq=1000, num=10):
    freqs = array_analyze_freq_liner(get_nyquist_freq(freq))
    y_pos, y_label = get_ticks_label_set(freqs[::-1], num)
    return y_pos, y_label


def align_digits(labels, truncate: int = 2):
    labels = [math.floor(d * 10 ** truncate) / (10 ** truncate) for d in labels]
    return labels


def main():
    freq = 100
    sample_rate = freq_to_sample_rate(freq)

    bandwidth = 1.5
    center_frequency = 1.0

    cwtmat, freq_rate = cwt(generate_signal(freq), get_scales(sample_rate),
                            get_cmor_wavelet_type(bandwidth, center_frequency))

    x_pos, x_label = get_xpos_xlabel(freq, 10)
    y_pos, y_label = get_ypos_ylabel(freq, 10)

    x_label = align_digits(x_label, 1)
    y_label = align_digits(y_label, 2)

    plt.yticks(y_pos, y_label)
    plt.xticks(x_pos, x_label)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")

    plt.imshow(np.abs(cwtmat), aspect='auto')


if __name__ == "__main__":
    main()
