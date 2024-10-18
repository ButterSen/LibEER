import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import yaml

from utils.store import save_state
from utils.metric import Metric
from data_utils.preprocess import ele_normalize
import torch.nn.functional as F
import math

param_path = 'config/model_param/MSMDA.yaml'

dim_feature = 310

class MSMDA(nn.Module):
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3, number_of_source = 14, pretrained=False):
        
        super(MSMDA, self).__init__()
        global dim_feature
        dim_feature = num_electrodes * in_channels
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        for i in range(number_of_source):
            '''
            这行代码的作用是根据类名动态创建一个类的实例，并将其赋值给相应的变量。
            通过执行 =DSFE()，可以创建一个新的 DSFE 类的实例，并将其赋值给 self.DSFE{i} 变量。
            '''
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(num_classes) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor
            # to extract the domain specific feature of target data
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            # discrepency loss
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(
                pred_src, dim=1), label_src.long().squeeze())

            return cls_loss, mmd_loss, disc_loss

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))
            return pred


def pretrained_CFE(pretrained=False):
    model = CFE()
    if pretrained:
        pass
    return model

class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        if dim_feature == 160:
            self.module = nn.Sequential(
                nn.Linear(dim_feature, 128),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(128, 64),
                # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(64, 64),
                # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
        else:
            self.module = nn.Sequential(
                nn.Linear(dim_feature, 256),
                # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(256, 128),
                # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(128, 64),
                # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )

    def forward(self, x):
        x = self.module(x)
        return x

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 32),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x