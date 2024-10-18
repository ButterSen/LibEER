from models.Models import Model
from config.setting import preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
from models.DGCNN import NewSparseL2Regularization
import torch
import torch.optim as optim
import torch.nn as nn

# run this file with
#    python CDCN_train.py -model CDCN -epochs 300 -seed 2024 -batch_size 64 -sessions 1 2 3 -onehot
#    -seed 2024 0.8510/0.0880
#    python CDCN_train.py -model CDCN -dataset deap -dataset_path '/date1/yss/data/DEAP数据集/data_preprocessed_python' -setting deap_sub_dependent_10fold_setting -feature_type de_lds -bounds 5 5  -epochs 300 -batch_size 64 -seed 2 -sr 10 -label_used valence -onehot >CDCN/deap_valence_seed2.log
#    0.9230/0.1133
#    python CDCN_train.py -model CDCN -dataset deap -dataset_path '/date1/yss/data/DEAP数据集/data_preprocessed_python' -setting deap_sub_dependent_10fold_setting -feature_type de_lds -bounds 5 5  -epochs 300 -batch_size 64 -seed 1 -sr 10 -label_used arousal -onehot
#    0.9203/0.1220

#    seed dep
#    python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting  -dataset seed_de_lds -batch_size 256 -onehot -epochs 300 -seed 2024 >CDCN/b256e300.log
#    0.6823/0.2035	0.6376/0.2449
#    seediv dep
#    CUDA_VISIBLE_DEVICES=2 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -time_window 1 -feature_type de_lds -batch_size 64 -onehot -epochs 300 -seed 2024 >CDCN/s4_b64e300.log
#    0.5226/0.2197 0.4526/0.2300

#    hci dep
#    arousal
#    python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 16 -epochs 300 -lr 0.01  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 -onehot >CDCN/hci_arousal_b16e300lr0.01.log
#    valence
#    python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 300 -lr 0.01  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 -onehot >CDCN/hci_valence_b128e300lr0.01.log
#    both
#    python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 64 -epochs 300 -lr 0.01  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >CDCN/hci_both_b64e300lr0.01.log

#    deap dep
#    valence
#    python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 -onehot  >CDCN/deap_arousal_b256e300lr0.001.log
#    0.5771/0.1472	0.5341/0.1550
#    arousal
#    python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 -onehot  >CDCN/deap_arousal_b16e300lr0.001.log
#    0.6337/0.1418	0.5394/0.1376
#    both
#    python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >CDCN/deap_both_b16e300lr0.001.log
#    0.3808/0.1552	0.2890/0.1315

#    hci indep
#    arousal
#    CUDA_VISIBLE_DEVICES=1 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 16 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 -onehot  >CDCN_indep/hci_arousal_b16e300lr0.001.log
#    0.5593	0.5515
#    valence
#    CUDA_VISIBLE_DEVICES=1 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 64 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 -onehot >CDCN_indep/hci_valence_b64e300lr0.001.log
#    0.6769	0.6267
#    CUDA_VISIBLE_DEVICES=1 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >CDCN_indep/hci_both_b256e300lr0.001.log
#    0.3009	0.2471


#    deap indep
#    arousal
#    CUDA_VISIBLE_DEVICES=3 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 16 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 -onehot  >CDCN_indep/deap_arousal_b16e300lr0.001.log
#    0.4973	0.4917
#    valence
#    CUDA_VISIBLE_DEVICES=3 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 128 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 -onehot >CDCN_indep/deap_valence_b128e300lr0.001.log
#    0.5778	0.5772
#    both
#    CUDA_VISIBLE_DEVICES=3 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 128 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >CDCN_indep/deap_both_b128e300lr0.001.log
#    0.239	0.2258

#    seed indep
#    CUDA_VISIBLE_DEVICES=2 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 128 -epochs 300 -time_window 1 -feature_type de_lds -seed 2024 -onehot >CDCN_indep/s4_b128e300.log
#    0.5772	0.5866
#    seediv indep
#    CUDA_VISIBLE_DEVICES=2 nohup python CDCN_train.py -metrics 'acc' 'macro-f1' -model CDCN -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 256 -epochs 300 -time_window 1 -feature_type de_lds -seed 2024 -onehot >CDCN_indep/s4_b256e300.log
#    0.3103	0.2701

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
            # model to train
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            model = Model['CDCN'](channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005, eps=0.0001)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "CDCN")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                 output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
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
