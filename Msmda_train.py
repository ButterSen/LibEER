from models.Models import Model
from config.setting import seed_sub_independent_leave_one_out_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.MsMdaTraining import train
from models.DGCNN import NewSparseL2Regularization
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_utils.preprocess import ele_normalize

# run this file with
#    python Msmda_reproduction.py -sessions 1 2 3 -model MS-MDA -batch_size 256 -epochs 200 -lr 0.01 -setting  'seed_sub_independent_leave_one_out_setting' -seed 20 -sr 15
#    0.9397
# python Msmda_train.py -batch_size 256 -epochs 200 -lr 0.01 -setting
def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    setting.onehot = False
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
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, keep_dim=True)

            datasets_train = []
            if len(val_data) == 0:
                val_data = test_data.copy()
                val_label = test_label.copy()
            samples_source = len(train_data[0])
            for j in range(len(train_data)):
                train_label[j] = np.array(train_label[j])
                train_data[j] = ele_normalize(np.array(train_data[j]))
                # print(train_data[j].shape)
                valid_elements = (train_data[j].shape[0] // samples_source) * samples_source
                if valid_elements != train_data[j].shape[0]:
                    train_data[j] = train_data[j][:valid_elements]
                    train_label[j] = train_label[j][:valid_elements]
                train_data[j] = train_data[j].reshape(samples_source, -1)
                # print(train_data[j].shape, train_label[j].shape)
                if train_label[j].shape[0]!=0:
                    datasets_train.append(torch.utils.data.TensorDataset(torch.Tensor(train_data[j]), torch.Tensor(train_label[j])))

            def trans(o_data, o_label):
                o_label[0] = np.array(o_label[0])
                o_data[0] = ele_normalize(np.array(o_data[0]))
                length = (o_data[0].shape[0] // samples_source) * samples_source
                if length != o_data[0].shape[0]:
                    o_data[0] = o_data[0][0][:length]
                    o_label[0] = o_label[0][:length]
                o_data[0] = o_data[0].reshape(samples_source, -1)
                return o_data, o_label

            test_data, test_label = trans(test_data, test_label)
            val_data, val_label = trans(val_data, val_label)

            # samples_target = len(test_data[0])
            # test_label = np.array(test_label[0])
            #
            # test_data = ele_normalize(np.array(test_data[0]))
            # test_data = test_data.reshape(samples_target, -1)
            # print(len(train_data))
            # model to train


            model = Model['MsMda'](channels, feature_dim, num_classes, number_of_source=samples_source)
            # Train one round using the train one round function defined in the model
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data[0]), torch.Tensor(val_label[0]))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data[0]), torch.Tensor(test_label[0]))

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "MsMda")
            round_metric = train(model=model, datasets_train=datasets_train, dataset_val=dataset_val, dataset_test=dataset_test, output_dir=output_dir, samples_source=samples_source, device=device, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                 batch_size=args.batch_size, epochs=args.epochs, criterion=criterion)
            best_metrics.append(round_metric)
    # best metrics: every round metrics dict
    # subjects metrics: (subject, sub_round_metric)
    result_log(args, best_metrics)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # log out train state
    state_log(args)
    main(args)
