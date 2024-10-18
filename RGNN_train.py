from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.graphTraining import train
from models.RGNN_official import SparseL1Regularization
from data_utils.constants.seed import SEED_RGNN_ADJACENCY_MATRIX
from data_utils.constants.deap import DEAP_RGNN_ADJACENCY_MATRIX
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
from torch.utils.data import Dataset

# run this file with
#    python RGNN_train.py -onehot -batch_size 16 -lr 0.002 -sessions 1 2 -epochs 80 -setting seed_sub_dependent_front_back_setting -seed 2
#    0.8466/0.1074

#    seediv dep
#    python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 150 -time_window 1 -feature_type de_lds -seed 2024 >RGNN/s4_b16e150.log
#    0.4540/0.2290	0.3824/0.2309

#    seed dep
#    python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_de_lds -batch_size 32 -epochs 150 -time_window 1 -feature_type de_lds -seed 2024 >RGNN/b32e150lr0.001.log
#    0.7655/0.1692	0.7252/0.2008

#    hci dep
#    valence
#    CUDA_VISIBLE_DEVICES=3 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 >RGNN/hci_valence_b256e150lr0.005.log
#    0.6486/0.1736	0.5041/0.2034
#    arousal
#    CUDA_VISIBLE_DEVICES=3 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 64 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >RGNN/hci_arousal_b64e150lr0.01.log
#    0.7096/0.1979	0.5766/0.2539
#    both
#    CUDA_VISIBLE_DEVICES=3 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >RGNN/hci_both_b256e150lr0.005.log
#    0.4946/0.2320	0.3597/0.2384


#    deap dep
#    deap arousal
#    python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >RGNN/deap_arousal_b512e150lr0.01.log
#    0.6609/0.1391	0.4927/0.1286
#    deap valence
#    python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 >RGNN/deap_valence_b256e150lr0.01.log
#    0.5590/0.1624	0.4725/0.1755
#    deap both
#    python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >RGNN/deap_both_b256e150lr0.01.log
#    0.4453/0.1435	0.2588/0.1208

#    seed indep
#    python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_de_lds -batch_size 32 -epochs 150 -time_window 1 -feature_type de_lds -seed 2024 >RGNN_indep/b32e150.log
#    0.572	0.5137

#    seediv indep
#    python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 32 -epochs 150 -time_window 1 -feature_type de_lds -seed 2024 -lr 0.003 >RGNN_indep/s4_b32e150lr0.003.log
#    0.4413	0.433

#    hci indep
#    arousal
#    CUDA_VISIBLE_DEVICES=0 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >RGNN_indep/hci_arousal_b512e150lr0.005.log
#    0.5726	0.5689
#    valence
#    CUDA_VISIBLE_DEVICES=0 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 >RGNN_indep/hci_valence_b512e150lr0.005.log
#    0.6589	0.4433
#    both
#    CUDA_VISIBLE_DEVICES=0 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >RGNN_indep/hci_both_b256e150lr0.005.log
#    0.3743	0.196

#    deap indep
#    arousal
#    CUDA_VISIBLE_DEVICES=0 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >RGNN_indep/hci_arousal_b128e150lr0.005.log
#    0.4386	0.4046
#    valence
#    CUDA_VISIBLE_DEVICES=0 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 >RGNN_indep/hci_valence_b128e150lr0.01.log
#    0.5215	0.4488
#    both
#    CUDA_VISIBLE_DEVICES=0 nohup python RGNN_train.py -metrics 'acc' 'macro-f1' -model RGNN_official -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >RGNN_indep/hci_both_b512e150lr0.005.log
#    0.1909	0.1302


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
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            # model to train
            model = Model['RGNN_official'](channels, feature_dim, num_classes)
            # noise label
            train_label = torch.Tensor(model.noise_label(train_label))
            # train_label = F.log_softmax(train_label, dim=1)
            if args.dataset.startswith("seed"):
                edge_adj = torch.Tensor(SEED_RGNN_ADJACENCY_MATRIX)
            elif args.dataset.startswith("deap") or args.dataset.startswith("hci"):
                edge_adj = torch.Tensor(DEAP_RGNN_ADJACENCY_MATRIX)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
            # criterion = nn.KLDivLoss(reduction='sum')
            criterion = nn.CrossEntropyLoss()
            loss_func = SparseL1Regularization(0.01)
            output_dir = make_output_dir(args, "RGNN")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, edge_adj=edge_adj, device=device, output_dir=output_dir, metrics=args.metrics, optimizer=optimizer,
                                 batch_size=args.batch_size, epochs=args.epochs, criterion=criterion, loss_func=loss_func, loss_param=model.edge_weight)
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
