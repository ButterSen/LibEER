from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
from models.GCBNet_BLS import SparseL2Regularization
import torch
import torch.optim as optim
import torch.nn as nn


# run this file with
#    python GCBNet_BLS_reproduction.py -onehot -batch_size 16 -lr 0.001 -sessions 1 2 -seed 2024
#

#    seed dep
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_de_lds -batch_size 16 -epochs 150 -lr 0.0015 -seed 2024 -onehot >GCBNet_BLS/b16e150lr0.0015.log
#    0.7664/0.1744	0.7252/0.2120
#    seediv dep
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 8 -epochs 150 -lr 0.001 -time_window 1 -feature_type de_lds -seed 2024 -onehot >GCBNet_BLS/s4_b8e150lr0.001.log
#    0.5351/0.2245	0.4691/0.2246


#    hci dep
#    arousal
#    CUDA_VISIBLE_DEVICES=1 nohup python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >GCBNet_BLS/hci_arousal_b256e150lr0.01.log
#    0.6960/0.2209	0.5778/0.2718
#    valence
#    0.7060/0.2105	0.5896/0.2745
#    both
#    CUDA_VISIBLE_DEVICES=1 nohup python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >GCBNet_BLS/hci_both_b256e150lr0.005.log
#    0.4924/0.2797	0.3668/0.2725


#    deap dep
#    arousal
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 -onehot >GCBNet_BLS/deap_arousal_b512e150lr0.005.log
#    0.6107/0.1656	0.5043/0.1636
#    valence
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 -onehot >GCBNet_BLS/deap_valence_b256e150lr0.005.log
#    0.5702/0.1507	0.5140/0.1725
#    both
#    CUDA_VISIBLE_DEVICES=2 nohup python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >GCBNet_BLS/deap_both_b256e150lr0.005.log
#    0.3751/0.1529	0.2700/0.1352


#    seed indep
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_de_lds -batch_size 32 -epochs 150 -lr 0.001 -seed 2024 >GCBNet_BLS_indep/b32e150lr0.001.log
#    56.32%	0.5143
#    seeeiv indep
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 64 -epochs 150 -lr 0.0015 -time_window 1 -feature_type de_lds -seed 2024 >GCBNet_BLS_indep/s4_b64e150lr0.0015.log
#    0.4054	0.4273

#    hci indep
#    arousal
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >GCBNet_BLS_indep/hci_arousal_b256e150lr0.01.log
#    0.6085	0.5626
#    valence
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >GCBNet_BLS_indep/hci_arousal_b512e150lr0.005.log
#    0.7106	0.6183
#    both
#    python GCBNet_BLS_train.py -metrics 'acc' 'macro-f1' -model GCBNet_BLS -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >GCBNet_BLS_indep/hci_both_b256e150lr0.005.log
#    0.3692	0.3299

#    deap indep
#    arousal
#    CUDA_VISIBLE_DEVICES=2 nohup python GCBNet_train.py -metrics 'acc' 'macro-f1' -model GCBNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >GCBNet_indep/deap_arousal_b512e150lr0.01.log
#    0.5059	0.4638
#    valence
#    CUDA_VISIBLE_DEVICES=2 nohup python GCBNet_train.py -metrics 'acc' 'macro-f1' -model GCBNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 128 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 >GCBNet_indep/deap_valence_b128e150lr0.005.log
#    0.5234	0.4903
#    both
#    CUDA_VISIBLE_DEVICES=2 nohup python GCBNet_train.py -metrics 'acc' 'macro-f1' -model GCBNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >GCBNet_indep/deap_both_b256e150lr0.005.log
#    0.2049	0.1841



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
            # print(len(train_data))
            # model to train

            model = Model['GCBNet_BLS'](channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
            criterion = nn.CrossEntropyLoss()
            loss_func = SparseL2Regularization(0.001).to(device)
            output_dir = make_output_dir(args, "GCBNet_BLS")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_test=dataset_test, dataset_val=dataset_val, device=device,
                                 output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                 batch_size=args.batch_size, epochs=args.epochs, criterion=criterion, loss_func=loss_func, loss_param=model.fc.weight)
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
