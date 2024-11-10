import argparse
import os.path
import time
from pathlib import Path

import torch

def make_output_dir(args, model):
    output_dir = Path(args.output_dir)
    output_dir = output_dir / model
    if args.setting is not None:
        output_dir = output_dir / args.setting
    else:
        output_dir = output_dir / args.experiment_mode
        output_dir = output_dir / args.split_type
    if args.label_used is not None:
        if len(args.label_used) == 1:
            output_dir = output_dir/args.label_used[0]
        else:
            output_dir = output_dir/ "both".join(label for label in args.label_used)
    return output_dir

def save_state(output_dir, model, optimizer, epoch, r_idx='last', rr_idx='last', metric=None, state='best'):
    # compatibility
    if type(output_dir) is argparse.Namespace:
        output_dir = make_output_dir(output_dir, output_dir.model)
    else:
        output_dir = Path(output_dir)
    if not ( r_idx == 'last' and rr_idx == 'last'):
        output_dir = output_dir / str(r_idx)
        output_dir = output_dir / str(rr_idx)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"An error occurred: {e.strerror}")
    checkpoint_path = output_dir / f'checkpoint-{str(epoch)}' if metric is None \
        else output_dir / f'checkpoint-{state}{metric}'
    save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(save, checkpoint_path)
    print(f"save model to {checkpoint_path}")


def save_data(args, data, label):
    save_dir = Path(args.data_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mode = {
        'subject-dependent': 'sub-dep',
        'subject-independent': 'sub-In',
        'cross-session': 'cro-sess'
    }
    save_path = save_dir / f'{args.dataset}'
    save_path = save_path / f'{args.feature_type}-tw-{args.time_window}ol-{args.overlap}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Saving Processed Data To {save_path}")



def save_res(args, metric):
    log_dir = Path(args.log_dir)
    add_dir(log_dir)
    log_file = log_dir / time.strftime("%Y-%m-%d %H:%M:%S", args.time)
    if not os.path.exists(log_file):
        f = open(log_file, 'w')
        f.write(str(args))
        f.close()
    f = open(log_file, 'a')
    f.write('\n'+str(metric))
    f.close()


def add_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
