from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
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
# CUDA_VISIBLE_DEVICES=2  nohup python EEGNet_train.py -setting seed_sub_dependent_front_back_setting  -dataset seed_raw -onehot -batch_size 512 -sample_length 200 -stride 200 -only_seg -lr 0.001 -sessions 1 2 -epochs 200 > EEGNet/repro512_0001.log &


#    seed dep
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -sample_length 200 -stride 200 -only_seg -batch_size 256 -epochs 100 -seed 2024 -lr 0.001 -onehot >EEGNet/b256e100l001.log
#    0.5881/0.1622	0.5441/0.1759
#    seed iv dep
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 512 -epochs 150 -seed 2024 -sample_length 200 -stride 200 -only_seg -onehot >EEGNet/s4_b512e150l001.log
#    0.2989/0.1353	0.2659/0.1358


#    hci dep
#    arousal
#    CUDA_VISIBLE_DEVICES=2 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >EEGNet/hci_arousal_b256e300lr0.04.log
#    0.6742/0.2171	0.5450/0.2005
#    valence
#    CUDA_VISIBLE_DEVICES=2 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >EEGNet/hci_valence_b128e300lr0.04.log
#    0.6115/0.1676	0.5035/0.1728
#    both
#    CUDA_VISIBLE_DEVICES=2 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >EEGNet/hci_both_b128e300lr0.04.log
#    0.3832/0.1951	0.2456/0.1371

#    deap dep
#    arousal
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.02 -only_seg -sample_length 512 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 -onehot >EEGNet/deap_arousal_b512e300lr0.02.log
#    0.6130/0.1588	0.5326/0.1305
#    valence
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 -onehot >EEGNet/deap_valence_b256e300lr0.02.log
#    0.5150/0.1157	0.4785/0.1170
#    both
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 -onehot >EEGNet/deap_both_b256e300lr0.04.log
#    0.3941/0.1153	0.2919/0.1021

#    seed iv indep
#    python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 128 -epochs 150 -seed 2024 -sample_length 200 -stride 200 -only_seg >EEGNet_indep/s4_b128e150.log &
#    28.19%	0.2835


#    hci indep
#    valence
#    CUDA_VISIBLE_DEVICES=1 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 300 -lr 0.04 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >EEGNet_indep/hci_valence_b256e300lr0.04.log
#    0.5706	0.5383
#    arousal
#    CUDA_VISIBLE_DEVICES=1 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 512 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >EEGNet_indep/hci_arousal_b512e300lr0.02.log
#    0.547	0.5402
#    both
#    CUDA_VISIBLE_DEVICES=1 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 256 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >EEGNet_indep/hci_both_b256e300lr0.02.log
#    0.3484	0.2796


#    deap indep
#    valence
#    CUDA_VISIBLE_DEVICES=3 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 128 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >EEGNet_indep/deap_valence_b128e300lr0.02.log
#    0.5236	0.4974
#    arousal
#    CUDA_VISIBLE_DEVICES=3 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >EEGNet_indep/deap_arousal_b512e300lr0.02.log
#    0.4894	0.4894
#    both
#    CUDA_VISIBLE_DEVICES=3 nohup python EEGNet_train.py -metrics 'acc' 'macro-f1' -model EEGNet -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 300 -lr 0.02 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >EEGNet_indep/deap_both_b256e300lr0.02.log
#    0.2541	0.2444


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

            model = Model['EEGNet'](channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "EEGNet")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device
                                 ,output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
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
