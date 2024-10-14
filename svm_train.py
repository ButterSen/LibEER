import numpy as np

from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.metric import Metric
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
import torch
import torch.optim as optim
import torch.nn as nn


def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    best_metrics = []
    subjects_metrics = [[]for _ in range(len(data))]
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            # split train and test data by specified experiment mode
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)
            # print(len(train_data))
            # model to train
            train_label = np.argmax(train_label, axis=1)
            test_label = np.argmax(test_label, axis=1)
            val_label = np.argmax(val_label, axis=1)
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            model = Model['svm'](channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            train_data = train_data.reshape(train_data.shape[0], -1)
            test_data = test_data.reshape(test_data.shape[0], -1)
            model.svc.fit(train_data, train_label)
            pred = model.svc.predict(test_data)
            metric = Metric(args.metrics)
            metric.update(pred, test_label)
            metric.value()
            round_metric = metric.values
            # output_dir = make_output_dir(args, "svm")
            for m in args.metrics:
                print(f"best_test_{m}: {round_metric[m]:.2f}")
            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx-1].append(round_metric)
    # best metrics: every round metrics dict
    # subjects metrics: (subject, sub_round_metric)
    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # log out train state
    state_log(args)
    main(args)
