from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.BiDANNTraining import train

import torch
import torch.optim as optim
import torch.nn as nn

# run this file with
#    reproduction
#    python BiDANN_train.py -onehot -sample_length 9 -batch_size 16 -lr 0.00004 -sessions 1 2 -epochs 80 -setting seed_sub_dependent_front_back_setting
#    0.8929/0.0874




def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
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
            print(f"train data shape: {train_data.shape}, train label shape: {train_label.shape}")
            print(f"train_label[0]: {train_label[0]}")
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            model = Model['BiDANN'](channels, feature_dim, num_classes,sample_length=9, 
                 domain_classes = 2,lambda_ = 0.5, device=device)
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
            f_criterion = nn.CrossEntropyLoss()
            left_ld_criterion = nn.CrossEntropyLoss()
            right_ld_criterion = nn.CrossEntropyLoss()
            global_criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "BiDANN")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                 output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                 batch_size=args.batch_size, epochs=args.epochs, f_criterion=f_criterion, left_ld_criterion=left_ld_criterion,
                                 right_ld_criterion=right_ld_criterion, global_criterion=global_criterion,
                                 loss_func=None, loss_param=model)
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
    main(args)
   