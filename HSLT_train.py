from torch.optim.lr_scheduler import CosineAnnealingLR

from models.Models import Model
from config.setting import deap_sub_independent_leave_one_out_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
import torch
import torch.optim as optim
import torch.nn as nn

# run this file with
# CUDA_VISIBLE_DEVICES=1  nohup python HSLT_train.py -time_window 6 -overlap 3 -feature_type psd_lds -dataset deap -dataset_path '/date1/yss/data/DEAP数据集/data_preprocessed_python' -bounds 4 6 -batch_size 512 -lr 0.01 -epochs 100 -label_used valence arousal  -metrics acc weighted-f1 ck -onehot >HSLT/repro_512_001_va_lds.log &
#
# CUDA_VISIBLE_DEVICES=1  nohup python HSLT_train.py -time_window 6 -overlap 3 -feature_type psd_lds -dataset deap -dataset_path '/date1/yss/data/DEAP数据集/data_preprocessed_python' -bounds 4 6 -batch_size 512 -lr 0.01 -epochs 100 -label_used valence  -metrics acc weighted-f1 ck -onehot >HSLT/repro_512_001_v_lds.log &
# acc : 0.6918/0.0906 f1 : 0.6598/0.1038
# CUDA_VISIBLE_DEVICES=1  nohup python HSLT_train.py -time_window 6 -overlap 3 -feature_type psd_lds -dataset deap -dataset_path '/date1/yss/data/DEAP数据集/data_preprocessed_python' -bounds 4 6 -batch_size 512 -lr 0.01 -epochs 100 -label_used arousal  -metrics acc weighted-f1 ck -onehot >HSLT/repro_512_001_a_lds.log &
# acc : 0.6881/0.1157 f1 : 0.6596/0.1139

#    hci dep
#    arousal
#    CUDA_VISIBLE_DEVICES=2 nohup python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 100 -lr 0.005 -time_window 1 -feature_type psd -bounds 5 5 -label_used arousal -seed 2024 -onehot >HSLT/hci_arousal_b256e100lr0.005.log
#    0.6774/0.1722	0.5842/0.1950
#    valence
#    CUDA_VISIBLE_DEVICES=2 nohup python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 100 -lr 0.005 -time_window 1 -feature_type psd -bounds 5 5 -label_used valence -seed 2024 -onehot >HSLT/hci_valence_b256e100lr0.005.log
#    0.6400/0.1140	0.5577/0.1312
#    both
#    CUDA_VISIBLE_DEVICES=2 nohup python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 100 -lr 0.01 -time_window 1 -feature_type psd -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >HSLT/hci_both_b256e100lr0.01.log
#    0.4699/0.2076	0.3476/0.1969

#    deap dep
#    arousal
#    python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 100 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >HSLT/deap_both_b256e100lr0.005.log
#    0.5974/0.1882	0.5010/0.1808
#    valence
#    python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 100 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 -onehot >HSLT/deap_valence_b512e100lr0.01.log
#    0.5620/0.1814	0.4821/0.1873

#    seed iv dep
#    python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 150 -time_window 1 -feature_type de_lds -onehot -seed 2024 >HSLT/s4_b16e150.log
#    0.4028/0.2380	0.3092/0.2447

#    seed dep
#    python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_de_lds -batch_size 512 -epochs 100 -onehot -seed 2024 -lr 0.005 >HSLT/b512e100lr_0.005.log
#    0.6483/0.2047	0.5882/0.2336

#    seed indep
#    python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_de_lds -batch_size 512 -epochs 100 -onehot -seed 2024 -lr 0.01 >HSLT_indep/b512e100_lr0.01.log
#    0.56	0.5575

#    seediv indep
#    CUDA_VISIBLE_DEVICES=2 nohup python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 150 -time_window 1 -feature_type de_lds -onehot -seed 2024 >HSLT_indep/s4_b16e150.log
#    0.3033	0.1164

#    deap indep
#    valence
#    python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 64 -epochs 100 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 -onehot >HSLT_indep/deap_valence_b64e100lr0.01.log
#    0.6694	0.6222
#    arousal
#    python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 100 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 -onehot >HSLT_indep/deap_arousal_b256e100lr0.005.log
#    0.4948	0.4743
#    CUDA_VISIBLE_DEVICES=1 nohup python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 100 -lr 0.005 -time_window 1 -feature_type psd -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >HSLT_indep/hci_both_b512e100lr0.005.log
#    0.1731	0.1687

#    hci indep
#    arousal
#    CUDA_VISIBLE_DEVICES=1 nohup python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 100 -lr 0.005 -time_window 1 -feature_type psd -bounds 5 5 -label_used arousal -seed 2024 -onehot >HSLT_indep/hci_arousal_b256e100lr0.005.log
#    0.4948	0.4743
#    valence
#    CUDA_VISIBLE_DEVICES=1 nohup python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 64 -epochs 100 -lr 0.01 -time_window 1 -feature_type psd -bounds 5 5 -label_used valence -seed 2024 -onehot >HSLT_indep/hci_valence_b64e100lr0.01.log
#    0.6694	0.6222
#    both
#    CUDA_VISIBLE_DEVICES=1 nohup python HSLT_train.py -metrics 'acc' 'macro-f1' -model HSLT -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 100 -lr 0.005 -time_window 1 -feature_type psd -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >HSLT_indep/hci_both_b128e100lr0.005.log#
#    0.3507	0.2719

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
    subjects_metrics = [[] for _ in range(len(data))]
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
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            # model to train

            model = Model['HSLT'](channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
            if num_classes > 2:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.BCELoss()
            output_dir = make_output_dir(args, "HSLT")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                 output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer, scheduler=scheduler,
                                 batch_size=args.batch_size, epochs=args.epochs, criterion=criterion)
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
