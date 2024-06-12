from scipy.signal import find_peaks
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def sec2samp(sec, sr):
    """
    This function converts seconds to samples according to the sampling rate
    :param sec: (float) seconds to convert to sample
    :param sr: (float) sampling rate
    :return samp: (float) samples
    """
    return int(sec * sr)


def samp2sec(samp, sr):
    """
    This function converts samples to seconds according to the sampling rate
    :param samp: (float) sample to convert to seconds
    :param sr: (float) sampling rate
    :return sec: (float) seconds
    """
    return samp * (1/sr)


def filter_peaks(peaks, dist_samp=1000):
    """
    This function filters out peaks that are too close to each other and keeps only the first one of a bout of peaks.
    This is handy for audio signal processing, to extract the onset of a specific sound within a periodic sound sample
    :param peaks: (numpy array) sample numbere of each of the detected peak
    :param dist_samp: (int) minimum interval between 2 peaks
    :return: filtered_peaks (numpy array) indices of all the first peaks with the minimal dist_samp in between
    """
    filtered_peaks = []
    if len(peaks) == 0:
        return filtered_peaks

    # Initialize the first peak
    filtered_peaks.append(peaks[0])

    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] >= dist_samp:
            filtered_peaks.append(peak)

    return filtered_peaks


def find_clicks(data, sr, thresh, distance_s=0.5):
    """
    This function extracts the peaks from an audio signals. You need to define a threshold as well as the minimal
    distance between the peaks (in seconds)
    :param data: (numpy array)
    :param sr: (float) sampling frequency of the audio signal
    :param thresh: (float) threshold above which to consider peaks
    :param distance_s: (float) minimal distance between peaks in seconds
    :return:
    """
    if len(data.shape) > 1:
        data = data[:, 0]
    # Binarize the signal based on the specified threshold:
    data_bin = (data > thresh).astype(float)

    # Find the peaks:
    peaks, _ = find_peaks(data_bin, height=0.5)

    # Remove peaks that are too close from each other:
    filtered_peaks = filter_peaks(peaks, dist_samp=sec2samp(distance_s, sr))

    # Convert the peaks to time stamps:
    peaks_sec = [samp2sec(x, sr) for x in filtered_peaks]

    return filtered_peaks, peaks_sec
