from data_utils.constants.deap import DEAP_CHANNEL_NAME
from data_utils.constants.seed import SEED_CHANNEL_NAME
from models.Models import Model
from models.TSception import generate_TS_channel_order
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
import numpy as np


# run this file with
# deap batch 64 hci batch 32


#    seed dep
#    CUDA_VISIBLE_DEVICES=0 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -batch_size 16 -epochs 200 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception/b16e200.log
#    0.6401/0.1644	0.6053/0.1851
#    seed iv dep
#    CUDA_VISIBLE_DEVICES=1 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 300 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception/s4_b16e300.log
#    0.3606/0.1512	0.3277/0.1508

#    seed indep
#    CUDA_VISIBLE_DEVICES=1 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -batch_size 32 -epochs 200 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception_indep/b32e200.log
#    0.456	0.4354
#    seed iv indep
#    CUDA_VISIBLE_DEVICES=3 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 300 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception_indep/s4_b16e300.log
#    0.3419	0.2683

#    deap indep
#    valence
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 64 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception_indep/deap_valence_b64e300lr0.001.log
#    0.5444	0.4894
#    arousal
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.001.log
#    0.459	0.4556
#    both
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.002.log
#    0.2464	0.2324

#   hci indep
#   valence
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception/hci_valence_b32e300lr0.001.log
#   0.5736	0.5476
#   arousal
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception_indep/hci_arousal_b32e300lr0.001.log
#   0.523	0.5025
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >TSception_indep/hci_both_b32e300lr0.001.log
#   0.2699	0.2195

#   hci dep
#   arousal
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.002.log
#   0.6826/0.2310	0.5629/0.2379
#   valence
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception/hci_valence_b32e300lr0.002.log
#   0.6112/0.1552	0.5051/0.1669
#   both
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >TSception/hci_both_b32e300lr0.002.log
#   0.4000/0.2060	0.2719/0.1365

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
            if args.dataset.startswith('hci'):
                model = Model['TSception'](channels, feature_dim, num_classes, inception_window=[0.25, 0.125, 0.0625])
            else:
                model = Model['TSception'](channels, feature_dim, num_classes)

            indexes = np.array([])
            if args.dataset == "deap" or args.dataset == "hci":
                indexes = generate_TS_channel_order(DEAP_CHANNEL_NAME)
            elif args.dataset.startswith("seed"):
                indexes = generate_TS_channel_order(SEED_CHANNEL_NAME)
            train_data = train_data[:, indexes, :]
            val_data = val_data[:, indexes, :]
            test_data = test_data[:, indexes, :]

            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "TSception")
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
