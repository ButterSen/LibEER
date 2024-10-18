from data_utils.preprocess import normalize
from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.DBNTraining import train
from models.DGCNN import NewSparseL2Regularization
import torch
import torch.optim as optim
import torch.nn as nn

# run this file with
#    reproduction
#    seed
#    under supervise lr 0.02 python DBN_train.py -sessions 1 2 -batch_size 512 -epochs 200
#    0.8118/0.0813

#    seed dep
#    CUDA_VISIBLE_DEVICES=1 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset seed_de_lds -batch_size 512 -epochs 100 -seed 2024 >DBN/b512e100.log
#    0.7188/0.1902	0.6739/0.2281

#    seed iv dep
#    CUDA_VISIBLE_DEVICES=2 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset seediv_raw -batch_size 512 -epochs 100 -time_window 1 -feature_type de_lds -seed 2024
#    0.4556/0.2119	0.3761/0.2068

#    hci dep
#    arousal
#    CUDA_VISIBLE_DEVICES=0 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 300 -lr 0.01  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024   >DBN/hci_arousal_b128e300lr0.01.log
#    0.6851/0.2172	0.5718/0.2663
#    valence
#    CUDA_VISIBLE_DEVICES=0 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 300 -lr 0.01  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024  >DBN/hci_valence_b128e300lr0.01.log
#    0.6203/0.2490	0.5284/0.2704
#    both
#    CUDA_VISIBLE_DEVICES=0 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 64 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024  >DBN/hci_both_b64e300lr0.001.log
#    0.4438/0.2678  0.3196/0.2787

#    deap dep
#    arousal
#    CUDA_VISIBLE_DEVICES=2 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 100 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024 >DBN/deap_arousal_b256e100lr0.001.log
#    0.6460/0.1942	0.5261/0.1985
#    valence
#    CUDA_VISIBLE_DEVICES=2 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 256 -epochs 100 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024 >DBN/deap_valence_b256e100lr0.001.log
#    0.5608/0.1738	0.4861/0.1933
#    both
#    CUDA_VISIBLE_DEVICES=2 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting deap_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 512 -epochs 100 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024 >DBN/deap_both_b512e100lr0.001.log
#    0.3950/0.1399	0.2488/0.1079

#    seed indep
#    python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset seed_de_lds -batch_size 512 -epochs 100 -seed 2024
#    0.3616	0.2267
#    seed iv indep
#    python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 256 -epochs 100 -time_window 1 -feature_type de_lds -seed 2024
#    0.3682	0.326

#    hci indep
#    arousal
#    CUDA_VISIBLE_DEVICES=3 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 16 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024   >DBN_indep/hci_arousal_b16e300lr0.001.log
#    0.575	0.5624
#    valence
#    CUDA_VISIBLE_DEVICES=3 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 64 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024  >DBN_indep/hci_valence_b64e300lr0.001.log
#    0.6927	0.6551
#    both
#    CUDA_VISIBLE_DEVICES=3 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 128 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence arousal -seed 2024  >DBN_indep/hci_both_b128e300lr0.001.log
#    0.283	0.2758

#    deap indep
#    arousal
#    CUDA_VISIBLE_DEVICES=0 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 64 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used arousal -seed 2024   >DBN_indep/deap_arousal_b64e300lr0.001.log
#    0.5099	0.5009
#    valence
#    CUDA_VISIBLE_DEVICES=0 nohup python DBN_train.py -metrics 'acc' 'macro-f1' -model DBN -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 128 -epochs 300 -lr 0.001  -time_window 1 -feature_type de_lds -bounds 5 5 -label_used valence -seed 2024  >DBN_indep/deap_valence_b128e300lr0.001.log
#    0.5502	0.5314
#    both
#    0.2784	0.2436



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
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample', method="minmax")

            model = Model['DBN'](channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "DBN")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,
                                 output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose,
                                 batch_size=args.batch_size, epochs=args.epochs)
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
