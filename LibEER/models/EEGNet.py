import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import yaml

from utils.store import save_state
from utils.metric import Metric

from data_utils.preprocess import normalize

param_path = 'config/model_param/EEGNet.yaml'

class EEGNet(nn.Module):
    def __init__(self, num_electrodes=62, datapoints=128, num_classes=3, F1=8, D=2, dropout=0.5):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, datapoints//2), padding='same', bias=False)
        self.BN1 = nn.BatchNorm2d(self.F1)
        # self.depth_conv = nn.Conv2d(in_channels=self.F1, out_channels=self.F1 * self.D, kernel_size=(num_electrodes, 1), bias=False,
        #                             groups=self.F1)
        self.depth_conv = Conv2dWithConstraint(in_channels=self.F1, out_channels=self.F1 * self.D, kernel_size=(num_electrodes, 1), bias=False,
                                    groups=self.F1)
        self.BN2 = nn.BatchNorm2d(self.D * self.F1)
        self.act1 = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=4)
        self.dropout1 = nn.Dropout(dropout)
        self.sep_conv = nn.ModuleList()
        self.sep_conv.append(
            nn.Conv2d(in_channels=self.D * self.F1, out_channels=self.D * self.F1, kernel_size=(1, 16), padding='same', bias=False,
                      groups=self.D * self.F1))
        F2 = self.D * self.F1
        self.sep_conv.append(nn.Conv2d(in_channels=self.D * self.F1, out_channels=F2, kernel_size=1, bias=False))
        self.BN3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=8)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(F2 * (datapoints // 32), num_classes)

    def get_param(self):
        return

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.depth_conv.weight)
        nn.init.kaiming_normal_(self.sep_conv[0].weight)
        nn.init.kaiming_normal_(self.sep_conv[1].weight)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    def weight_constraint(self, parameters, min_value, max_value):
        for param in parameters:
            param.data.clamp_(min_value, max_value)

    def forward(self, x):
        # x shape -> (batch_size, channels, datapoints)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.depth_conv(x)
        x = self.BN2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.sep_conv[0](x)
        x = self.sep_conv[1](x)
        x = self.BN3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Conv2dWithConstraint(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, max_value=1.0, bias=False, groups=1):
        super(Conv2dWithConstraint, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.max_value = max_value

    def forward(self, x):
        output = self.conv(x)
        output = torch.clamp(output, max=self.max_value)
        return output