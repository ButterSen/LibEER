import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.store import save_data
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split
import random

# def train_test_split(data, label, setting):
#     """
#     Provides division of training set and test set under various experimental settings No matter how the experimental
#     settings are, they are all based on trail division, so trail is a basic division unit. For the three typical
#     experimental settings on a dataset, subject-dependent, subject-independent, cross-session,they can be operated
#     based on each subject’s trail, each subject, each session
#           input : all the eeg data and label which can directly be taken as an input
#           output : data and label that make up the training set or test set
#           input shape -> data :   (session, subject, trail, sample, sample_length, time_window, channel, band_feature)
#                          label :  (session, subject, trail, sample, label)
#           output shape -> data :  (sample, sample_length, time_window, channel, band_feature)
#                           label : (sample, label)
#     """
#     train_data = []
#     train_label = []
#     test_data = []
#     test_label = []
#     if setting.experiment_mode == "subject-dependent":
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for session in data for subject in session for i in setting.train_part for sample in
#                       subject[i - 1]]
#         train_label = [sample for session in label for subject in session for i in setting.train_part for sample in
#                        subject[i - 1]]
#
#         test_part = list(set(range(1, len(data[0][0]) + 1)) - set(setting.train_part))
#
#         test_data = [sample for session in data for subject in session for i in test_part for sample in
#                      subject[i - 1]]
#         test_label = [sample for session in label for subject in session for i in test_part for sample in
#                       subject[i - 1]]
#
#     elif setting.experiment_mode == "subject-independent":
#
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for session in data for i in setting.train_part for trail in session[i-1]
#                       for sample in trail]
#         train_label = [sample for session in label for i in setting.train_part for trail in session[i-1]
#                        for sample in trail]
#
#         test_part = list(set(range(1, len(data[0]) + 1)) - set(setting.train_part))
#
#         test_data = [sample for session in data for i in test_part for trail in session[i-1] for sample in trail]
#         test_label = [sample for session in label for i in test_part for trail in session[i-1] for sample in trail]
#
#     elif setting.experiment_mode == "cross-session":
#
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for i in setting.train_part for session in data[i-1] for subject in session for trail in
#                       subject for sample in trail]
#         train_label = [sample for i in setting.train_part for session in label[i-1] for subject in session for trail in
#                        subject for sample in trail]
#
#         test_part = list(set(range(1, len(data) + 1)) - set(setting.train_part))
#
#         test_data = [sample for i in test_part for subject in data[i-1] for trail in subject for sample in trail]
#         test_label = [sample for i in test_part for subject in label[i-1] for trail in subject for sample in trail]
#
#     train_data = np.asarray(train_data)
#     train_label = np.asarray(train_label)
#     test_data = np.asarray(test_data)
#     test_label = np.asarray(test_label)
#     if setting.normalize:
#         for i in range(len(train_data[0][0])):
#             scaler = StandardScaler()
#             train_data[:, :, i] = scaler.fit_transform(train_data[:, :, i])
#             test_data[:, :, i] = scaler.transform(test_data[:, :, i])
#     # if setting.save_data:
#     #     save_data(train_data, train_label, test_data, test_label)
#     return train_data, train_label, test_data, test_label

def index_to_data(data, label, train_indexes, test_indexes, val_indexes, keep_dim=False):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    if keep_dim:
        for train_index in train_indexes:
            train_data.append(data[train_index])
            train_label.append(label[train_index])
        for test_index in test_indexes:
            test_data.append(data[test_index])
            test_label.append(label[test_index])
        if val_indexes[0] != -1:
            for val_index in val_indexes:
                val_data.append(data[val_index])
                val_label.append(label[val_index])
    else:
        for train_index in train_indexes:
            train_data.extend(data[train_index])
            train_label.extend(label[train_index])
        for test_index in test_indexes:
            test_data.extend(data[test_index])
            test_label.extend(label[test_index])
        if val_indexes[0] != -1:
            for val_index in val_indexes:
                val_data.extend(data[val_index])
                val_label.extend(label[val_index])
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_label = np.array(train_label)
        test_label = np.array(test_label)
        val_data = np.array(val_data)
        val_label = np.array(val_label)
    return train_data, train_label, val_data, val_label, test_data, test_label


def get_split_index(data, label, setting=None):
    tts = {}
    if setting.split_type == "kfold":
        kf = KFold(setting.fold_num, shuffle=True if setting.fold_shuffle == 'true' or setting.fold_shuffle == 'True' else False,
                   random_state=setting.seed if setting.fold_shuffle == 'true' else None)
        tts['train'] = [list(train_index) for train_index, _ in kf.split(label)]
        tts['test'] = [list(test_index) for _, test_index in kf.split(label)]
    elif setting.split_type == "leave-one-out":
        loo = LeaveOneOut()
        tts['train'] = [list(train_index) for train_index, _ in loo.split(label)]
        tts['test'] = [list(test_index) for _, test_index in loo.split(label)]
    elif setting.split_type == "front-back":
        if setting.front >= len(label):
            print(f"using front-back split type and {setting.experiment_mode} experiment mode")
            print(f"front size {setting.front} > split part num {len(label)}")
            print("please check your experiment mode or split type")
            exit(1)
        tts['train'] = [[i for i in range(setting.front)]]
        tts['test'] = [[setting.front + i for i in range(len(label) - setting.front)]]
    elif setting.split_type == "early-stop":
        if setting.experiment_mode == "subject-dependent":
            # data need to be split balanced
            # input data : [[not-repetitive] * trails], label : [[repetitive] * trails]
            # output : split index
            tts['test'] = [[]]
            tts['train'] = [[]]
            tts['val'] = [[]]
            groups = {}
            for index, value in enumerate(label):
                if isinstance(value[0], np.ndarray):
                    value_key = tuple(value[0])
                else:
                    value_key = value[0]
                if value_key in groups:
                    groups[value_key].append(index)
                else:
                    groups[value_key] = [index]
            # print(groups)
            others = []
            for indexes in groups.values():
                random.shuffle(indexes)
                total_length = len(indexes)
                test_num = int(setting.test_size * total_length)
                val_num = int(setting.val_size * total_length)
                train_num = int((1-setting.test_size-setting.val_size)*total_length)
                tts['test'][0].extend(indexes[:test_num])
                tts['val'][0].extend(indexes[test_num:test_num+val_num])
                tts['train'][0].extend(indexes[test_num+val_num:test_num+val_num+train_num])
                others.extend(indexes[test_num+val_num+train_num:])
            if len(others) != 0:
                random.shuffle(others)
                expect_test_num = int(len(label) * setting.test_size)
                expect_val_num = int(len(label) * setting.val_size)
                test_num = expect_test_num - len(tts['test'][0])
                val_num = expect_val_num - len(tts['val'][0])
                tts['test'][0].extend(others[:test_num])
                tts['val'][0].extend(others[test_num:test_num+val_num])
                tts['train'][0].extend(others[test_num+val_num:])
        else:
            tts['test'] = [[]]
            tts['train'] = [[]]
            tts['val'] = [[]]
            indexes = [i for i in range(len(label))]
            random.shuffle(indexes)
            total_length = len(indexes)
            test_num = int(setting.test_size * total_length)
            val_num = int(setting.val_size * total_length)
            train_num = total_length - test_num - val_num
            tts['test'][0].extend(indexes[:test_num])
            tts['val'][0].extend(indexes[test_num:test_num + val_num])
            tts['train'][0].extend(indexes[test_num + val_num:])
    else:
        print("wrong split type, please check out")
        exit(1)
    assert setting.sr is None or (max(setting.sr)<=len(label) and min(setting.sr) > 0), \
        "secondary rounds out of limit or secondary rounds set less than 0"
    if setting.sr is not None:
        tts['train'] = [tts['train'][i-1] for i in setting.sr]
        tts['test'] = [tts['test'][i-1] for i in setting.sr]
        if 'val' in tts:
            tts['val'] = [tts['val'][i-1] for i in setting.sr]
    if 'val' not in tts:
        tts['val'] = [[-1] for _ in tts['train']]
    return tts


def merge_to_part(data, label, setting=None):
    """
    According to experiment mode, merge (session, subject, trail, sample) to (corresponding_part, sample)
    :param data: -> (session, subject, trail, sample, ...)
    :param label: -> (session, subject, trail, sample, ...)
    :param setting: -> setting for dataset process
    setting.experiment_mode: choices->["subject-dependent", "subject-independent", "cross-session"]
    setting.sessions: which sessions we choose to use, index start from 1, default is all
    :return: if not subject-dependent:
                 data: -> (corresponding_part, sample)
                 label: ->(corresponding_part, sample)
             else if subject-dependent and cross-trail:
                 data: -> (subject, trail, sample)
                 label: -> (subject, trail, sample)
                 else subject-dependent and not cross-trail"
                 data: -> (subject, , sample)
                 label: -> (subject, , sample)
    """
    assert setting.sessions is None or (max(setting.sessions)<=len(label) and min(setting.sessions) >= 0), \
        "sessions set fault, session not exist in dataset"
    if setting.sessions is None:
        sessions = range(len(data))
    else:
        sessions = [i - 1 for i in setting.sessions]
    m_data = []
    m_label = []
    if setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'true':

        m_data = [[] for _ in range(len(data[0]) * len(sessions))]
        m_label = [[] for _ in range(len(data[0]) * len(sessions))]
        for i in sessions:
            for idx1, subject in enumerate(data[i]):
                for idx2, trail in enumerate(subject):
                    m_data[i*len(data[i])+idx1].append(trail)
        for i in sessions:
            for idx1, subject in enumerate(label[i]):
                for idx2, trail in enumerate(subject):
                    m_label[i*len(data[i])+idx1].append(trail)
    elif setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'false':
        m_data = [[] for _ in range(len(data[0]))]
        m_label = [[] for _ in range(len(data[0]))]
        for i in sessions:
            for idx1, subject in enumerate(data[i]):
                for idx2, trail in enumerate(subject):
                    for sample in trail:
                        m_data[i*len(data[0])+idx1].append([sample])
        for i in sessions:
            for idx1, subject in enumerate(label[i]):
                for idx2, trail in enumerate(subject):
                    for sample in trail:
                        m_label[i*len(data[0])+idx1].append([sample])
    elif setting.experiment_mode == "subject-independent":
        m_data = [[[] for _ in range(len(data[0]))]]
        m_label = [[[] for _ in range(len(data[0]))]]
        for i in sessions:
            for idx, subject in enumerate(data[i]):
                for trail in subject:
                    m_data[0][idx].extend(trail)
        for i in sessions:
            for idx, subject in enumerate(label[i]):
                for trail in subject:
                    m_label[0][idx].extend(trail)
    elif setting.experiment_mode == "cross-session":
        m_data = [[[] for _ in range(len(sessions))]]
        m_label = [[[] for _ in range(len(sessions))]]
        for i in sessions:
            for subject in data[i]:
                for trail in subject:
                    m_data[0][i].extend(trail)
        for i in sessions:
            for subject in label[i]:
                for trail in subject:
                    m_label[0][i].extend(trail)
    assert setting.pr is None or (max(setting.pr)<=len(m_label) and min(setting.pr) > 0), \
        "primary rounds out of limit or primary rounds set less than 0"
    if setting.pr is not None:
        m_data = [m_data[i-1] for i in setting.pr]
        m_label = [m_label[i-1] for i in setting.pr]
    return m_data, m_label
