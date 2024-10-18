import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from tqdm import tqdm
import yaml

from utils.store import save_state
from utils.metric import Metric

param_path = '../config/model_param/CDCN.yaml'


class CDCN(nn.Module):
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3, growth_rate=12, block_layers=None,
                 dropout=0.4):
        # num_electrodes(int): The number of electrodes.
        # in_channels(int): The feature dimension of each electrode.
        # num_classes(int): The number of classes to predict.
        # growth_rate(int): The number of additional feature maps generated per layer
        # block_layers: The number of convolution blocks of each denseblock
        super(CDCN, self).__init__()

        if block_layers is None:
            block_layers = [6, 6, 6]
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.block_layers = block_layers
        self.dropout = dropout
        self.get_params()
        self.init_weight()

        self.features = nn.Sequential(nn.Conv2d(1, 2 * self.growth_rate, kernel_size=(1, 5), bias=False),
                                      nn.BatchNorm2d(2 * self.growth_rate),
                                      nn.ReLU()
                                      )

        num_features_maps = 2 * self.growth_rate
        for i, num_layers in enumerate(self.block_layers):
            block = DenseBlock(num_layers, num_features_maps, self.growth_rate)

            if i == 0:
                self.block_tran = nn.Sequential(block)
            else:
                self.block_tran.add_module("denseblock%d" % (i + 1), block)

            num_features_maps += num_layers * self.growth_rate

            if i == 0:
                transition = Transition(num_features_maps, num_features_maps)
                self.block_tran.add_module("transition%d" % (i + 1), transition)
            elif i == 1:
                transition = Transition(num_features_maps, num_features_maps, True)
                self.block_tran.add_module("transition%d" % (i + 1), transition)

        self.trail = nn.Sequential(nn.BatchNorm2d(num_features_maps),
                                   nn.ReLU())

        self.GAP = GlobalAveragePooling()
        self.fc = nn.Linear(num_features_maps, self.num_classes)

    def get_params(self):
        try:
            fd = open(param_path, 'r', encoding='utf-8')
            model_param = yaml.load(fd, Loader=yaml.FullLoader)
            fd.close()
            self.growth_rate = model_param['params']['growth_rate']
            self.block_layers = model_param['params']['block_layers']
            self.dropout = model_param['params']['dropout']
            print("\nUsing setting from {}\n".format(param_path))
        except IOError:
            print("\n{} may not exist or not available".format(param_path))

        print("CDCN Model, Parameters:\n")
        print("{:45}{:20}".format("self.growth_rate:", self.growth_rate))
        print("{:60}{:20}".format("block_layers:",  ', '.join(str(layer) for layer in self.block_layers)))
        print("{:45}{:20}\n".format("dropout rate:", self.dropout))
        if self.growth_rate != 12 or self.block_layers != [6, 6, 6] or self.dropout != 0.5:
            print("Not Using Default Setting, the performance may be not the best")
        print("Starting......")

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.block_tran(x)
        x = self.trail(x)
        x = self.GAP(x)
        x = F.dropout(x, p=self.dropout)
        x = self.fc(x)
        return x



class Convblock(nn.Sequential):
    def __init__(self, num_inputs_features, growth_rate):
        super(Convblock, self).__init__()
        self.bn = nn.BatchNorm2d(num_inputs_features)
        self.relu = nn.ReLU()
        self.pad = nn.ZeroPad2d((0, 0, 1, 1))
        self.conv = nn.Conv2d(num_inputs_features, growth_rate, kernel_size=(3, 1), stride=1, padding=0, bias=False)

    def forward(self, x):
        output = self.bn(x)
        output = self.relu(output)
        output = self.pad(output)
        output = self.conv(output)
        output = F.dropout(output, p=0.5)
        return torch.cat([x, output], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_inputs_features, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            if i == 0:
                self.layer = nn.Sequential(
                    Convblock(num_inputs_features + i * growth_rate, growth_rate)
                )
            else:
                layer = Convblock(num_inputs_features + i * growth_rate, growth_rate)
                self.layer.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, x):
        x = self.layer(x)
        return x


class Transition(nn.Sequential):
    def __init__(self, num_inputs_features, num_outputs_features, if_pad=False):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_inputs_features)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(num_inputs_features, num_outputs_features, (1, 1), stride=1)
        self.pad = nn.ZeroPad2d((0, 0, 0, 1))
        self.pool = nn.MaxPool2d((2, 1), stride=2)
        self.if_pad = if_pad


    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        if self.if_pad:
            x = self.pad(x)
        x = self.pool(x)
        return x


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2, 3))
