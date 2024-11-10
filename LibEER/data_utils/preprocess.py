import torch.nn.functional as F

import numpy as np
import scipy.signal
from scipy import signal
from scipy.signal import filtfilt, stft

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from functools import partial


# 对eeg信号进行各项数据预处理操作（去除眼动干扰，带通滤波，提取频段，分段样本，提取特征, 归一化等）
# 并且同时
# 最终希望处理成为能够经过划分后就能输入模型的数据


def preprocess(data, baseline, sample_rate, pass_band, extract_bands, time_window, overlap, car=False, whiten=False,
               sample_length=1, stride=1, only_seg=False, feature_type='DE', eog_clean=True, normalization=False):
    """
    Provide preprocessing operations
    input shape -> data:  (session, subject, trail, channel, original_data)
                   label: (session, subject, trail, label)
    output shape -> data :  (session, subject, trail, sample, time, channel, feature)
                    label : (session, subject, trail, sample, label)
    """
    if baseline is not None:
        data = baseline_removal(data, baseline)
    if not only_seg:
        if pass_band != [-1, -1]:
            data = bandpass_filter(data, sample_rate, pass_band)
        if eog_clean:
            data = eog_remove(data)
        # data, label = frequency_band_extraction(data, label, sample_rate, extract_bands, time_window)
        data = feature_extraction(data, sample_rate, extract_bands, time_window, overlap, feature_type)
    data, feature_dim = segment_data(data, sample_length, stride)
    return data, feature_dim
def noise_label(train_label, num_classes=3, level=0.1):
    if type(train_label[0]) is np.ndarray:
        train_label = [np.where(tl==1)[0] for tl in train_label]

    noised_label = [[] for _ in train_label]
    if num_classes == 4:
        for i, label in enumerate(train_label):
            if label == 0:
                noised_label[i] = [1 - 3 / 4 * level, 1 / 4 * level, 1 / 4 * level, 1 / 4 * level]
            elif label == 1:
                noised_label[i] = [1 / 3 * level, 1 - 2 / 3 * level, 1 / 3 * level, 0]
            elif label == 2:
                noised_label[i] = [1 / 4 * level, 1 / 4 * level, 1 - 3 / 4 * level, 1 / 4 * level]
            else:
                noised_label[i] = [1 / 3 * level, 0, 1 / 3 * level, 1 - 2 / 3 * level]
    elif num_classes == 3:
        for i, label in enumerate(train_label):
            if label == 0:
                noised_label[i] = [1 - 2 / 3 * level, 2 / 3 * level, 0]
            elif label == 1:
                noised_label[i] = [1 / 3 * level, 1 - 2 / 3 * level, 1 / 3 * level]
            else:
                noised_label[i] = [0, 2 / 3 * level, 1 - 2 / 3 * level]
    elif num_classes == 2:
        for i, label in enumerate(train_label):
            if label == 0:
                noised_label[i] = [1, 0]
            elif label == 1:
                noised_label[i] = [0, 1]
    return noised_label


def baseline_removal(data, base):
    for ses_i, ses_data in enumerate(data):
        for sub_i, sub_data in enumerate(ses_data):
            for trail_i, trail_data in enumerate(sub_data):
                trail_time = trail_data.shape[1]
                base_time = base[ses_i][sub_i][trail_i].shape[1]
                base_data = base[ses_i][sub_i][trail_i]
                for i in range(int(trail_time/base_time)):
                    trail_data[:,i*base_time:(i+1)*base_time] = trail_data[:,i*base_time:(i+1)*base_time] - base_data
                last = trail_time % base_time
                if last !=0 :
                    trail_data[:,-last:] = trail_data[:,-last:] - base_data[:,:last]
                data[ses_i][sub_i][trail_i] = trail_data
    return data


def bandpass_filter(data, frequency, pass_band):
    """
    Perform baseband filtering operation on EEG signal
    input: EEG signal
    output: EEG signal with band-pass filtering
    input shape : (session, subject, trail, channel, original_data)
    output shape : (session, subject, trail, channel, filter_data)
    """
    # define Nyquist frequency which is the minimum sampling rate defined to prevent signal aliasing
    nyq = 0.5 * frequency
    # get the coefficients of a Butterworth filter
    b, a = signal.butter(N=5, Wn=[pass_band[0] / nyq, pass_band[1] / nyq], btype='bandpass')
    # perform linear filtering
    # process on all channels
    for ses_i, ses_data in enumerate(data):
        for sub_i, sub_data in enumerate(ses_data):
            for trail_i, trail_data in enumerate(sub_data):
                data[ses_i][sub_i][trail_i] = \
                    filtfilt(b, a, trail_data)

    return data

def whiten(data):
    """
    whitening operation for data
    :param data:
    :return:
    """
    # centering operation
    new_data = []
    for session in range(len(data)):
        new_session = []
        for subject in range(len(data[0])):
            new_subject = []
            for trail in range(len(data[0][0])):
                trail = np.array(trail)
                # center operation
                trail_mean = trail.mean(axis=0)
                trail_center = trail - trail_mean
                # covariance matrix
                cov = np.dot(trail_center.T, trail_center) / (trail_center.shape[0])
                # eigenvalue computing
                eig_vals, eig_vecs = np.linalg.eigh(cov)
                D = np.diag(1.0 / np.sqrt(eig_vals))
                W = np.dot(eig_vecs, D).dot(eig_vecs.T)

                trail_whitened = np.dot(trail_center, W)
                new_subject.append(trail_whitened)
            new_session.append(new_subject)
        new_data.append(new_session)
    return new_data

# def ica_eog_remove(data):

def eog_remove(data):
    """
    Remove eye movement interference by artefact subspace reconstruction
    input: original eeg data
    output: eeg data with ocular artifacts removed
    input shape : (session, subject, trail, channel, filter_data)
    output shape : (session, subject, trail, channel, filter_data)
    """
    pca = PCA()
    return data


def feature_extraction(data, sample_rate, extract_bands, time_window, overlap, feature_type):
    """
    input: information processed after bandpass filter
    output:
    input shape -> data:  (session, subject, trail, channel, band, filter_data)
    output shape -> data:  (session, subject, trail, sample, channel, band, band_feature)
    """
    isLds = False
    if feature_type.endswith("_lds"):
        isLds = True
        feature_type = feature_type[:-4]
    fe = {
        'psd': psd_extraction,
        'de': de_extraction,
        'de_reduced': de_reduced_extraction
    }[feature_type]
    feature_data = []
    for ses_i, ses_data in enumerate(data):
        ses_fe = []
        for sub_i, sub_data in enumerate(ses_data):
            sub_fe = []
            for trail_i, trail_data in enumerate(sub_data):
                sub_fe_data = fe(trail_data, sample_rate, extract_bands, time_window, overlap)
                if isLds:
                    sub_fe_data = lds(sub_fe_data)
                # if trail_i == 0 and sub_i == 0:
                #     print(sub_fe_data)

                sub_fe.append(sub_fe_data)
            ses_fe.append(sub_fe)
        feature_data.append(ses_fe)
    return feature_data

def psd_extraction(data, sample_rate, extract_bands, time_window, overlap):
    """
    input shape -> data: (channel, filter_data)
    output shape -> data: (sample, channel, band_psd_feature)
    """
    if extract_bands is None:
        extract_bands = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]
    noverlap = int(overlap * sample_rate)
    window_size = int(time_window * sample_rate)
    if noverlap != 0:
        sample_num = (data.shape[1] - window_size) // (window_size-noverlap)
    else:
        sample_num = (data.shape[1]) // window_size
    psd_data = np.zeros((sample_num, data.shape[0], len(extract_bands)))
    t = 0
    for i in range(sample_num):
        f, psd = scipy.signal.welch(data[:, t:t+window_size],
                                    fs=sample_rate, nperseg=window_size, window='hamming')
        for b_i, bands in enumerate(extract_bands):
            psd_data[i, :, b_i] = np.mean(10 * np.log10(psd[:, bands[0]:bands[1]+1]), axis=1)
        t += window_size-noverlap
    return psd_data

def de_reduced_extraction(data, sample_rate, extract_bands, time_window, overlap):
    """
    use reduced operation to accelerate the DE feature extraction
    :param data: original eeg data, input shape: (channel, filter_data)
    :param sample_rate: sample rate of eeg signal
    :param extract_bands: the frequency bands that needs to be extracted
    :param time_window: time window of one extract part
    :param overlap: overlap
    :return: de feature need to be computed
    """
    if extract_bands is None:
        extract_bands = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]
    noverlap = int(overlap * sample_rate)
    window_size = int(time_window * sample_rate)
    if noverlap != 0:
        sample_num = (data.shape[1] - window_size) // (window_size-noverlap)
    else:
        sample_num = (data.shape[1]) // window_size
    de_data = np.zeros((sample_num, data.shape[0], len(extract_bands)))
    fs, ts, Zxx = stft(data, fs=sample_rate, window='hamming', nperseg=window_size, noverlap=noverlap, boundary=None)
    for b_idx, band in enumerate(extract_bands):
        fb_indices = np.where((fs >= band[0]) & (fs <= band[1]))[0]
        fourier_coe = np.real(Zxx[:, fb_indices, :])
        parseval_energy = np.mean(np.square(fourier_coe), axis=1)
        de_data[:,:,b_idx] = np.transpose(np.log2(100 * parseval_energy))[:sample_num]
    return de_data

def de_extraction(data, sample_rate, extract_bands, time_window, overlap):
    """
    DE feature extraction
    :param data: original eeg data, input shape: (channel, filter_data)
    :param sample_rate: sample rate of eeg signal
    :param extract_bands: the frequency bands that needs to be extracted
    :param time_window: time window of one extract part
    :param overlap: overlap
    :return: de feature need to be computed
    """
    if extract_bands is None:
        extract_bands = [[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]]
    nyq = 0.5 * sample_rate
    noverlap = int(overlap * sample_rate)
    window_size = int(time_window * sample_rate)
    if noverlap != 0:
        sample_num = (data.shape[1] - window_size) // (window_size - noverlap)
    else:
        sample_num = (data.shape[1]) // window_size
    de_data = np.zeros((sample_num, data.shape[0], len(extract_bands)))
    for b_idx, band in enumerate(extract_bands):
        b, a = signal.butter(3, [band[0]/nyq, band[1]/nyq], 'bandpass')
        band_data = signal.filtfilt(b, a, data)
        t = 0
        for i in range(sample_num):
            de_data[i,:,b_idx] = 1 / 2 * np.log2(2 * np.pi * np.e * np.var(band_data[:,t:t+window_size], axis=1, ddof=1))
            t += window_size-noverlap
    return de_data


def lds(data):
    """
    Process data using a linear dynamic system approach.

    :param data: Input data array with shape (time, channel, feature)
    :return: Transformed data with shape (time, channel, feature)
    """
    [num_t, num_channel, num_feature] = data.shape
    # Flatten the channel and feature dimensions
    data = data.reshape((data.shape[0], -1))

    # Initial parameters
    prior_correlation = 0.01
    transition_matrix = 1
    noise_correlation = 0.0001
    observation_matrix = 1
    observation_correlation = 1

    # Calculate the mean for initialization
    mean = np.mean(data, axis=0)
    data = data.T  # Transpose for easier manipulation of time dimension

    num_features, num_samples = data.shape
    P = np.zeros(data.shape)
    U = np.zeros(data.shape)
    K = np.zeros(data.shape)
    V = np.zeros(data.shape)

    # Initial Kalman filter setup
    K[:, 0] = prior_correlation * observation_matrix / (
                observation_matrix * prior_correlation * observation_matrix + observation_correlation) * np.ones(
        (num_features,))
    U[:, 0] = mean + K[:, 0] * (data[:, 0] - observation_matrix * prior_correlation)
    V[:, 0] = (np.ones((num_features,)) - K[:, 0] * observation_matrix) * prior_correlation

    # Apply the Kalman filter over time
    for i in range(1, num_samples):
        P[:, i - 1] = transition_matrix * V[:, i - 1] * transition_matrix + noise_correlation
        K[:, i] = P[:, i - 1] * observation_matrix / (
                    observation_matrix * P[:, i - 1] * observation_matrix + observation_correlation)
        U[:, i] = transition_matrix * U[:, i - 1] + K[:, i] * (
                    data[:, i] - observation_matrix * transition_matrix * U[:, i - 1])
        V[:, i] = (1 - K[:, i] * observation_matrix) * P[:, i - 1]

    # Return the processed data, reshaping it to match the original input shape
    return U.T.reshape((num_t, num_channel, num_feature))



def segment_data(data, sample_length, stride):
    """
    feature:
    input: original band features of EEG signal provided by SEED dataset
    output:
    input shape -> data:  (session, subject, trail, sample1, channel, band)
                    label: (session, subject, trail, sample1, label)
    output shape -> data:  (session, subject, trail, sample2, sample2_length, channel, band)
                    label: (session, subject, trail, sample2, label)
    raw_data:
    input: original data of EEG signal
    input shape -> data: (session, subject, trail, channel, data_points)
                   label: (session, subject, trail)
    output shape -> data: (session, subject, trail, sample, channel, seg_data_points)
                    label: (session, subject, trail)
    """
    if sample_length == 1:
        print(len(data[0][0]))
        print(len(data[0][0][0]))
        print(len(data[0][0][0][0]))
        return data, len(data[0][0][0][0][0])
    else:
        seg_data = []
        for ses_i, session in enumerate(data):
            seg_session = []
            for sub_i, subject in enumerate(data[ses_i]):
                seg_sub = []
                seg_sub_label = []
                for t_i, trail in enumerate(data[ses_i][sub_i]):
                    seg_trail = None
                    trail = np.array(trail)
                    if len(trail.shape) == 3:
                        # trail shape -> (sample, channel, band)
                        trail = np.asarray(trail)
                        num_sample = (len(trail) - sample_length) // stride + 1
                        seg_trail = np.zeros((num_sample, sample_length, len(trail[0]), len(trail[0][0])))
                        # Cutting a one-dimensional array through a sliding window to form a two-dimensional array
                        for i in range(num_sample):
                            seg_trail[i] = trail[i*stride:i*stride+sample_length]
                    elif len(trail.shape) == 2:
                        # trail shape -> (channel, data_points)
                        num_sample = (len(trail[0]) - sample_length) // stride + 1
                        seg_trail = np.zeros((num_sample, len(trail), sample_length))
                        for i in range(num_sample):
                            seg_trail[i] = trail[:, i*stride:i*stride+sample_length]
                    seg_sub.append(seg_trail)
                seg_session.append(seg_sub)
            seg_data.append(seg_session)
        if len(seg_data[0][0][0].shape) == 4:
            return seg_data, len(seg_data[0][0][0][0][0][0])
        elif len(seg_data[0][0][0].shape) == 3:
            return seg_data, len(seg_data[0][0][0][0][0])

def label_process(data, label, bounds=None, onehot=False, label_used=None):
    """
    input shape -> data: (session, subject, trail, sample)
                   label: (session, subject, trail)
    output shape -> data: (session, subject, trail, sample)
                    label: (session, subject, trail, sample)
    bounds shape -> 2, high emotion state > bounds[1], low emotion state < bounds[0]
    if dataset is hci, deap, dreamer, then label will be ordered by valence, arousal, dominance, liking
    """
    available_label = ['valence', 'arousal', 'dominance', 'liking']
    if label_used is None:
        label_used = ['valence']
    used_id = [available_label.index(item) for item in label_used]
    if type(label[0][0][0]) is np.ndarray:
        num_classes = np.power(2, len(used_id))
    else:
        num_classes = len(np.unique(label))
    new_label = []
    new_data = []
    for ses_i, ses_label in enumerate(label):
        new_ses_label = []
        new_ses_data = []
        for sub_i, sub_label in enumerate(ses_label):
            new_sub_label = []
            new_sub_data = []
            for trail_i, trail_label in enumerate(sub_label):
                new_trail_label = []
                new_trail_data = data[ses_i][sub_i][trail_i]
                num_sample = len(new_trail_data)
                if type(trail_label) is np.ndarray:
                    pro_label = []
                    for value_id in used_id:
                        value = trail_label[value_id]
                        if value <= bounds[0]:
                            pro_label.append(0)
                        elif value >= bounds[1]:
                            pro_label.append(1)
                    # pro_label shape -> (num_used_label, 2)
                    # processing into the ordinary labels
                    if len(pro_label) == len(used_id):
                        trail_label = int("".join(str(i)for i in pro_label),2)
                    else:
                        # discard the data and label
                        continue
                if onehot:
                    oh_code = np.zeros((1,num_classes), dtype='int32')
                    # print(trail_label)
                    oh_code[0][trail_label] = 1
                    trail_label = oh_code
                    new_trail_label = np.tile(trail_label, (num_sample, 1))
                else:
                    trail_label = np.ones(1, dtype='int32') * trail_label
                    new_trail_label = np.tile(trail_label, num_sample)
                new_sub_data.append(new_trail_data)
                new_sub_label.append(new_trail_label)
            new_ses_label.append(new_sub_label)
            new_ses_data.append(new_sub_data)
        new_label.append(new_ses_label)
        new_data.append(new_ses_data)

    return new_data, new_label, num_classes

def normalize(train_data, val_data, test_data=None, dim="sample", method="z-score"):

    all_data = np.concatenate((train_data, val_data), axis=0)
    data_shape = all_data.shape
    scaler = None
    scaled_test_data = None
    if dim == "sample":
        if len(data_shape) == 3:
            all_data = all_data.reshape(data_shape[0], data_shape[1] * data_shape[2])
        elif len(data_shape) == 4:
            all_data = all_data.reshape(data_shape[0], data_shape[1] * data_shape[2] * data_shape[3])
        scaled_data = None
        if method == "z-score":
            scaler  = StandardScaler()
            scaled_data = scaler.fit_transform(all_data)
        if method == "minmax":
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(all_data)
        if len(data_shape) == 3:
            scaled_data = scaled_data.reshape(data_shape[0], data_shape[1], data_shape[2])
        elif len(data_shape) == 4:
            scaled_data = scaled_data.reshape(data_shape[0], data_shape[1], data_shape[2], data_shape[3])
        if test_data is not None:
            if len(test_data.shape) == 3:
                test_data_reshaped = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
            elif len(test_data.shape) == 4:
                test_data_reshaped = test_data.reshape(test_data.shape[0],
                                                       test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

            scaled_test_data = scaler.transform(test_data_reshaped)

            if len(test_data.shape) == 3:
                scaled_test_data = scaled_test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2])
            elif len(test_data.shape) == 4:
                scaled_test_data = scaled_test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2],
                                                            test_data.shape[3])
        return scaled_data[:len(train_data)], scaled_data[len(train_data):], scaled_test_data
    if dim == "electrode":
        # data shape -> (sample, channel, band)
        all_data_t = all_data.reshape(len(all_data), len(all_data[0]) * len(all_data[0][0])).T
        test_data_t = test_data.reshape(len(test_data), len(test_data[0]) * len(test_data[0][0])).T
        for i in range(all_data_t.shape[0]):
            _range = np.max(all_data_t[i]) - np.min(all_data_t[i])
            all_data_t[i] = (all_data_t[i] - np.min(all_data_t[i])) / _range
            test_data_t[i] = (test_data_t[i] - np.min(all_data_t[i])) / _range
        norm_data = all_data_t.T.reshape(len(all_data), len(all_data[0]), len(all_data[0][0]))
        norm_test_data = test_data_t.T.reshape(len(test_data), len(test_data[0]), len(test_data[0][0]))
        return norm_data[:len(train_data)], norm_data[len(train_data):], norm_test_data

def ele_normalize(all_data):
    # data shape -> (sample, channel, band)
    all_data_t = all_data.reshape(len(all_data), len(all_data[0]) * len(all_data[0][0])).T
    for i in range(all_data_t.shape[0]):
        _range = np.max(all_data_t[i]) - np.min(all_data_t[i])
        all_data_t[i] = (all_data_t[i] - np.min(all_data_t[i])) / _range
    norm_data = all_data_t.T.reshape(len(all_data), len(all_data[0]), len(all_data[0][0]))
    return norm_data

def baseline_normalisation(data, baseline):
    # input shape : data (session, subject, trail)
    #               baseline (session, subject, trail)
    norm_data = []
    for ses_i in range(len(data)):
        session = data[ses_i]
        ses_base = baseline[ses_i]
        normal_session = []
        for sub_i in range(len(session)):
            subject = session[sub_i]
            sub_base = ses_base[sub_i]
            norm_subject = []
            for trail_i in range(len(subject)):
                trail = subject[trail_i]
                trail_base = sub_base[trail_i]
                norm_trail = []
                base_len = len(trail_base)
                for i in range(len(trail)//base_len):

                    norm_trail[i*base_len:(i+1)*base_len] = trail[i*base_len:(i+1)*base_len] / trail_base
                # if trail_i == 0 and sub_i == 0:
                #     print(norm_trail)
                norm_subject.append(norm_trail)
            normal_session.append(norm_subject)
        norm_data.append(normal_session)
    return norm_data


def generate_adjacency_matrix(channel_names, channel_adjacent):
    channel_names = np.array(channel_names)
    channel_num = len(channel_names)
    adjacency_matrix = np.zeros((channel_num, channel_num))
    for key, value in channel_adjacent.items():
        idx1 = np.where(channel_names == key)[0][0]
        for chan in value:
            idx2 = np.where(channel_names == chan)[0][0]
            adjacency_matrix[idx1][idx2] = 1
    return adjacency_matrix

def generate_rgnn_adjacency_matrix(channel_names, channel_loc, global_channel_pair):
    channel_names = np.array(channel_names)
    channel_num = len(channel_names)
    adjacency_matrix = np.zeros((channel_num, channel_num))
    for chan1 in channel_names:
        idx1 = np.where(channel_names == chan1)[0][0]
        for chan2 in channel_names:
            idx2 = np.where(channel_names == chan2)[0][0]
            if chan1 == chan2:
                adjacency_matrix[idx1][idx2] = 1
            else:
                cor1 = np.array(channel_loc[chan1])/10
                cor2 = np.array(channel_loc[chan2])/10
                dis_sq = 0
                for i in range(3):
                    dis_sq += np.square(cor1[i] - cor2[i])
                adjacency_matrix[idx1][idx2] = min(5/dis_sq, 1)
                adjacency_matrix[idx2][idx1] = min(5/dis_sq, 1)
    # print((np.where(adjacency_matrix > 0.1)[0].shape[0])/62/62)
    adjacency_matrix = differential_asymmetry_leverage(channel_names, adjacency_matrix, global_channel_pair)
    return adjacency_matrix
def differential_asymmetry_leverage(channel_names, adjacency_matrix, global_channel_pair):
    for pair in global_channel_pair:
        idx1 = np.where(channel_names == pair[0])[0][0]
        idx2 = np.where(channel_names == pair[1])[0][0]
        adjacency_matrix[idx1][idx2] -= 1
        adjacency_matrix[idx2][idx1] -= 1
    return adjacency_matrix






