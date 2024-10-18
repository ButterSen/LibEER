from models.Models import Model
from config.setting import deap_sub_dependent_10fold_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from data_utils.preprocess import normalize
from utils.args import get_args_parser
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
from utils.store import make_output_dir

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import numpy as np

# run this file with
#   CUDA_VISIBLE_DEVICES=3 nohup python ACRNN_train.py -setting deap_sub_dependent_10fold_setting -only_seg -cross_trail false -dataset_path '/date1/yss/data/DEAP数据集/data_preprocessed_python' -dataset deap -sample_length 384 -stride 384 -bounds 5 5 -fold_num 10 -fold_shuffle false -label_used valence -batch_size 10 -lr 10e-4 -epochs 200 -onehot > ACRNN/valence_reproduction.log
#   CUDA_VISIBLE_DEVICES=3 nohup python ACRNN_train.py -setting deap_sub_dependent_10fold_setting -only_seg -cross_trail false -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -sample_length 384 -stride 384 -bounds 5 5 -fold_num 10 -fold_shuffle false -label_used valence -batch_size 16 -lr 10e-4 -epochs 1000 -onehot > ACRNN/valence_reproduction.log

#   deap dep
#   arousal
#   CUDA_VISIBLE_DEVICES=2 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >ACRNN/deap_arousal_b16e1000lr0.001.log
#   0.6183/0.1432	0.4968/0.0920
#   valence
#   CUDA_VISIBLE_DEVICES=2 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >ACRNN/deap_valence_b16e1000lr0.001.log
#   0.5352/0.0929	0.4831/0.0777
#   both
#   CUDA_VISIBLE_DEVICES=2 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >ACRNN/deap_both_b16e1000lr0.001.log
#   0.3820/0.1134	0.2105/0.0613

#   hci dep
#   arousal
#   CUDA_VISIBLE_DEVICES=2 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci  -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >ACRNN/hci_arousal_b16e1000lr0.001.log
#   0.6626/0.2269	0.5517/0.2320
#   valence
#   CUDA_VISIBLE_DEVICES=2 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >ACRNN/hci_valence_b16e1000lr0.001.log
#   0.6051/0.1689	0.4939/0.1570
#   both
#   CUDA_VISIBLE_DEVICES=2 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >ACRNN/hci_both_b16e1000lr0.001.log
#   0.4100/0.2158	0.2710/0.1513

#   seed dep
#   python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -sample_length 200 -stride 200 -only_seg -lr 10e-4 -batch_size 16 -epochs 1000 -seed 2024 >ACRNN/b16e1000.log
#   0.4971/0.1315	0.4578/0.1418
#   seed iv dep
#   CUDA_VISIBLE_DEVICES=2 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >ACRNN/hci_both_b16e1000lr0.001.log#   0.2901/0.0710	0.1980/0.0542

#   seed indep
#   python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -sample_length 200 -stride 200 -only_seg -lr 10e-4 -batch_size 16 -epochs 1000 -seed 2024 >ACRNN_indep/b16e1000.log
#   0.4539	0.4237
#   seed iv indep
#
#   0.3197	0.1882

#   hci indep
#   valence
#   CUDA_VISIBLE_DEVICES=0 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >ACRNN_indep/hci_valence_b16e1000lr0.001.log
#   0.5453	0.5258
#   arousal
#   CUDA_VISIBLE_DEVICES=0 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci  -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >ACRNN_indep/hci_arousal_b16e1000lr0.001.log
#   0.5123	0.4945
#   CUDA_VISIBLE_DEVICES=0 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >ACRNN_indep/hci_both_b16e1000lr0.001.log
#   0.2721	0.2092

#   deap indep
#   valence
#   CUDA_VISIBLE_DEVICES=3 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >ACRNN_indep/deap_valence_b16e1000lr0.001.log
#   0.5194	0.4737
#   arousal
#   CUDA_VISIBLE_DEVICES=3 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >ACRNN_indep/deap_arousal_b16e1000lr0.001.log
#   0.4409	0.4151
#   both
#   CUDA_VISIBLE_DEVICES=3 nohup python ACRNN_train.py -metrics 'acc' 'macro-f1' -model ACRNN -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 1000 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >ACRNN_indep/deap_both_b16e1000lr0.001.log
#   0.2089	0.1516

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    # setting = deap_sub_dependent_10fold_setting(args)
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
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample')
            # print(train_data.shape, train_label.shape)
            train_data = np.transpose(train_data, (0, 2, 1))[:, np.newaxis, :, :]
            test_data = np.transpose(test_data, (0, 2, 1))[:, np.newaxis, :, :]
            val_data = np.transpose(val_data, (0, 2, 1))[:, np.newaxis, :, :]
            model = Model['ACRNN'](channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, gamma=0.3, step_size=100)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "ACRNN")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                 output_dir=output_dir,metrics=args.metrics, metric_choose=args.metric_choose,
                                 optimizer=optimizer, scheduler=scheduler, batch_size=args.batch_size, epochs=args.epochs, criterion=criterion)
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
