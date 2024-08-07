import argparse

import time
from config.setting import preset_setting
from data_utils.load_data import available_dataset


def get_args_parser():
    parser = argparse.ArgumentParser("EEG Lib for emotion recognition based on EEG", add_help=False)

    # training  parameters
    parser.add_argument('-batch_size', default=128, type=int, help='batch size per GPU')
    parser.add_argument('-epochs', default=40, type=int)
    parser.add_argument('-device', default='cuda', type=str, choices=['cuda', 'cpu'], help='which devices to train')
    parser.add_argument('-eval', default=False, action='store_true', help='if eval, perform evaluation only')
    parser.add_argument('-seed', default=1, type=int, help='random seed')
    parser.add_argument('-num_workers', default=4, type=int)
    parser.add_argument('-loss_func', default='crossEntropyLoss', type=str, help="the loss function")
    parser.add_argument('-metrics', default=['acc'], type=str, nargs='+', help='which metrics used to evaluate')
    parser.add_argument('-metric_choose', default='acc', type=str, help='which best metric choose to test')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-data_dir', default='./data_processed', type=str, help='the location to save processed data')

    # resume parameters
    parser.add_argument('-resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('-resume_epoch', default=0, type=int, help='resume epoch')
    parser.add_argument('-checkpoint', default=None, type=str, help='checkpoint')

    # model parameters
    parser.add_argument('-model', default='DGCNN', type=str, help='Name of model to train')

    # log parameters
    parser.add_argument('-log_dir', default='./log/', help='location of log dir')
    parser.add_argument('-output_dir', default='./result/', help='location of output dir')
    parser.add_argument('-time', default=time.localtime(), help='the time now')

    # preset parameters
    parser.add_argument('-setting', default='seed_sub_dependent_front_back_setting', choices=preset_setting,
                        help='using preset setting')

    # dataset parameters
    parser.add_argument('-dataset', default='seed_de_lds', type=str, choices=available_dataset,
                        help=f'available dataset are {available_dataset}')
    parser.add_argument('-dataset_path', default='/date1/yss/data/SEED数据集/SEED', type=str,
                        help='the location of dataset')
    parser.add_argument('-low_pass', default=0.3, type=float, help='the minimum frequency of bandpass filter')
    parser.add_argument('-high_pass', default=50, type=float, help='the maximum frequency of bandpass filter')
    parser.add_argument('-time_window', default=1, type=float, help='the num of sample points of preprocessing time window/s')
    parser.add_argument('-overlap', default=0, type=float, help='the length of overlap for each pretreatment/s')
    parser.add_argument('-sample_length', default=1, type=int, help='sequence length of each sample')
    parser.add_argument('-stride', default=1, type=int, help='the stride of a sliding window for data extraction')
    parser.add_argument('-feature_type', default='de_lds', type=str, help='the feature type need to compute')
    parser.add_argument('-eog_clean', default=False, action='store_true', help='whether clean eog')
    parser.add_argument('-only_seg', default=False, action='store_true', help='whether only segment data')
    parser.add_argument('-save_data', default=False, action='store_true', help='if save processed data')
    parser.add_argument('-normalize', default=True, )

    # train test split
    parser.add_argument('-cross_trail', default='true', type=str, help="whether use cross-trail setting")
    parser.add_argument('-experiment_mode', default='subject-dependent', type=str,
                        help='which experiment mode be selected')
    parser.add_argument('-split_type', default='front-back', type=str, choices=['kfold', 'leave-one-out', 'front-back'],
                        help="choose which method to split dataset")
    parser.add_argument('-fold_num', default=5, type=int, help='the number of folds')
    parser.add_argument('-fold_shuffle', default='true', type=str, help='whether shuffle when using k-fold split')
    parser.add_argument('-front', default=9, type=int, help='convert the first few data sets into training sets')
    parser.add_argument('-sessions', default=None, type=int, nargs='+', help="which sessions used to train")
    parser.add_argument('-test_size', default=0.2, type=float, help="the ratio of the test dataset")
    parser.add_argument('-val_size', default=0.2, type=float, help="the ratio of the val dataset")
    parser.add_argument('-pr',default=None, type=int, nargs='+', help="which primary rounds to train")
    parser.add_argument('-sr', default=None, type=int, nargs='+', help="which secondary rounds to train")
    parser.add_argument('-bounds', default=None, type=float, nargs='+', help="emotion score bounds:[low, high]")
    parser.add_argument('-onehot', default=True, action='store_true', help="if use onehot code")
    parser.add_argument('-label_used', default=None, type=str, nargs='+', help="valence, arousal, dominance, liking")
    parser.add_argument('-keep_dim',default=False, action='store_true')
    return parser
