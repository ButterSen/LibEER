import os

import numpy as np
import torch
import random

from utils.store import save_res


def state_log(args):
    log_dict = {
        "dataset": args.dataset,
        "feature type": args.feature_type,
        "model": args.model,
        "batch size": args.batch_size,
        "epochs": args.epochs,
        "learning rate": args.lr,
        "experiment mode": args.experiment_mode,
        "split_type": args.split_type,
        "log dir": args.log_dir,
        "output dir": args.output_dir,
    }
    print('_' * 43)
    for key, value in zip(log_dict.keys(), log_dict.values()):
        print("|{:^20}|{:^20}|".format(key, value))
    print('-' * 43)


def result_log(args, best_metrics):
    output = {}
    s = "|{:^10}|".format("Result")
    for metric_name in args.metrics:
        output[metric_name] = []
        s += "{:^10}|".format(metric_name)
    print(s)
    for idx, metric in enumerate(best_metrics):
        s_i = "|{:^10}|".format(idx + 1)
        for n in args.metrics:
            output[n].append(metric[n])
            s_i += "{:^10.3f}|".format(metric[n])
        print(s_i)
    for metric in args.metrics:
        print("ALLRound Mean and Std of {} : {:.4f}/{:.4f}".format(metric, np.mean(output[metric]), np.std(output[metric])))
        save_res(args, "ALLRound Mean and Std of {} : {:.4f}/{:.4f}".format(metric, np.mean(output[metric]), np.std(output[metric])))

def sub_result_log(args, subjects_metrics):
    sub_outputs = {}
    for i, sub_metric in enumerate(subjects_metrics):
        sub_output = {}
        for metric in args.metrics:
            sub_output[metric] = 0
            for r_metric in sub_metric:
                sub_output[metric] += r_metric[metric]
            sub_output[metric] /= len(subjects_metrics[i])
        sub_outputs[f"sub {i}"] = sub_output
    # sub_outputs: (sub, metric)
    save_res(args, sub_outputs)
    sub_mean_std = {}
    for metric in args.metrics:
        sub_metrics = []
        for sub_metric in sub_outputs.values():
            sub_metrics.append(sub_metric[metric])
        sub_mean_std[metric] = {"mean": np.mean(sub_metrics), "std": np.std(sub_metrics)}
        print("ALLRound Mean and Std of {} : {:.4f}/{:.4f}".format(metric, np.mean(sub_metrics), np.std(sub_metrics)))
        save_res(args,f"ALL Subjects {metric}: Mean: {np.mean(sub_metrics)}, Std: {np.std(sub_metrics)}")




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False