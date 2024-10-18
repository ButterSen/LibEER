from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
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
#    reproduction
#    python DGCNN_train.py -onehot -batch_size 16 -lr 0.0015 -sessions 1 2 -epochs 80 -setting seed_sub_dependent_front_back_setting
#    0.8948/0.0849

#    seed dep
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_de_lds -batch_size 32 -seed 2024 -epochs 80 -lr 0.0015 -onehot >DGCNN/b32_lr0.0015.log
#    0.8255/0.1561	0.7989/0.1893

#    seediv dep
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 32 -epochs 150 -time_window 1 -feature_type de_lds -seed 2024 -onehot >DGCNN/s4_b32e150.log
#    0.5239/0.2432	0.4594/0.2417

#    deap dep
#    both
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >DGCNN/deap_both_b512e150lr0.005.log
#    0.4186/0.1157	0.2912/0.1092
#    valence
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 -onehot >DGCNN/deap_valence_b512e150lr0.01.log
#    0.5607/0.1715	0.4908/0.1750
#    arousal
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 -onehot >DGCNN/deap_arousal_b512e150lr0.01.log
#    0.6268/0.1966	0.5394/0.2010

#    hci dep
#   `valence
#    CUDA_VISIBLE_DEVICES=1 nohup python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 >DGCNN/hci_valence_b256e150lr0.01.log
#    0.6783/0.2240	0.5478/0.2669
#    arousal
#    CUDA_VISIBLE_DEVICES=1 nohup python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >DGCNN/hci_arousal_b512e150lr0.01.log
#    0.6729/0.2773	0.5804/0.3141
#    both
#    CUDA_VISIBLE_DEVICES=1 nohup python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >DGCNN/hci_both_b256e150lr0.005.log
#    0.5306/0.2444	0.3975/0.2601


#    hci indep
#    arousal
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >DGCNN_indep/hci_arousal_b128e150lr0.005.log
#    0.5942	0.5702
#    valence
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 150 -lr 0.01 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 >DGCNN_indep/hci_valence_b512e150lr0.01.log
#    0.6319	0.5875
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 150 -lr 0.005 -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >DGCNN_indep/hci_both_b512e150lr0.005.log
#    0.4154	0.3808

#    seed indep
#    python DGCNN_train.py -metrics 'acc' 'macro-f1' -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path "/date1/yss/data/SEED数据集/SEED" -dataset seed_de_lds -batch_size 16 -seed 2024 -epochs 150 >DGCNN_indep/b16.log
#    0.6087	0.5722
#    seediv indep
#    CUDA_VISIBLE_DEVICES=2 nohup python DGCNN_train.py -metrics 'acc' 'macro-f1' -model DGCNN -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path "/date1/yss/data/SEED数据集/SEED_IV" -dataset seediv_raw -batch_size 32 -epochs 150 -time_window 1 -feature_type de_lds -seed 2024 -lr 0.0015 >DGCNN_indep/s4_b32e150_lr0.0015.log
#    0.4254	0.431


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

            model = Model['DGCNN'](channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
            criterion = nn.CrossEntropyLoss()
            loss_func = NewSparseL2Regularization(0.01).to(device)
            output_dir = make_output_dir(args, "DGCNN")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                 output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                 batch_size=args.batch_size, epochs=args.epochs, criterion=criterion, loss_func=loss_func, loss_param=model)
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
