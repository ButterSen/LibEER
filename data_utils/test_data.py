# import numpy as np
#
#
# # 定义一个函数来计算信号在特定频段内的能量
# def signal_energy_in_band(x, f_low, f_high):
#     # 计算信号的傅里叶变换
#     X = np.fft.fft(x)
#
#     # 计算信号在特定频段内的傅里叶系数
#     X_band = X[(f_low <= np.abs(np.fft.fftfreq(len(x)))) & (np.abs(np.fft.fftfreq(len(x))) <= f_high)]
#
#     # 计算信号在特定频段内的能量
#     signal_energy_in_band = np.sum(np.abs(X_band) ** 2)
#
#     return signal_energy_in_band
#
#
# # 生成一个长度为1024的正弦波
# x = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 1024))
#
# # 计算信号在特定频段内的能量
# signal_energy_in_band_x = signal_energy_in_band(x, 50, 150)
#
# # 计算傅里叶变换在特定频段内的能量
# X = np.fft.fft(x)
# fft_energy_in_band_X = np.sum(
#     np.abs(X[(50 <= np.abs(np.fft.fftfreq(len(x)))) & (np.abs(np.fft.fftfreq(len(x))) <= 150)]) ** 2)
#
# # 打印信号在特定频段内的能量和傅里叶变换在特定频段内的能量
# print("Signal energy in band:", signal_energy_in_band_x)
# print("FFT energy in band:", fft_energy_in_band_X)
# import numpy as np
# # import numpy as np
# # print(np.log2(2 * np.pi * np.e / 200)/2)
# # import random
# #
# # # 示例数据
# # groups = {
# #     'a': [1, 2, 3, 4, 5],
# #     'b': [6, 7, 8, 9, 10],
# #     'c': [11, 12, 13, 14, 15]
# # }
# #
# # # 随机打乱每个组的索引
# # for indexes in groups.values():
# #     random.shuffle(indexes)
# #
# # print(groups)
# from scipy.io import loadmat
# subject_data = loadmat("/date1/yss/data/SEED数据集/SEED_IV/eeg_feature_smooth/1/1_20160518.mat")
# keys = list(subject_data.keys())[3:]
# trail_datas = []
# for i in range(24):
#     trail_data = list(np.array(subject_data[]))
# # for i in sub
import load_data
data = [[][][]]
for ses_i, ses_data in enumerate(data):
    for sub_i, sub_data in enumerate(ses_data):
        for trail_i, trail_data in enumerate(sub_data):
            data[ses_i][sub_i][trail_i] = \
                filtfilt(b, a, tail_data)
load_data.read_hci("/data1/cxx/HCI数据集/")
