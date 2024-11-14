import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool
from torch_geometric.data import Data

import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.utils import scatter

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from data_utils.preprocess import normalize
from data_utils.constants.seed import SEED_RGNN_ADJACENCY_MATRIX
from data_utils.constants.deap import DEAP_RGNN_ADJACENCY_MATRIX

from utils.store import save_state
from utils.metric import Metric

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col
    inv_mask = ~mask
    loop_weight = torch.full(
        (num_nodes,),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    # allow negative edge weights
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        return norm.view(-1, 1) * x_j


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SymSimGCNNet(torch.nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, num_hiddens=[64], K=2,
                 dropout=0.7, domain_adaptation="", learn_edge_weight=True):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = torch.tensor(0)
        if num_nodes == 62:
            edge_weight = torch.Tensor(SEED_RGNN_ADJACENCY_MATRIX)
        elif num_nodes == 32:
            edge_weight = torch.Tensor(DEAP_RGNN_ADJACENCY_MATRIX)
        self.edge_index = edge_weight.to_sparse()._indices()
        self.edge_weight = nn.Parameter(edge_weight[self.xs, self.ys], requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens[0], K=K)
        self.fc = nn.Linear(num_hiddens[0], num_classes)
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = nn.Linear(num_hiddens[0], 2)

    def noise_label(self, train_label, level=0.1):
        if type(train_label[0]) is np.ndarray:
            train_label = [np.where(tl==1)[0] for tl in train_label]

        noised_label = [[] for _ in train_label]
        if self.num_classes == 4:
            for i, label in enumerate(train_label):
                if label == 0:
                    noised_label[i] = [1 - 3 / 4 * level, 1 / 4 * level, 1 / 4 * level, 1 / 4 * level]
                elif label == 1:
                    noised_label[i] = [1 / 3 * level, 1 - 2 / 3 * level, 1 / 3 * level, 0]
                elif label == 2:
                    noised_label[i] = [1 / 4 * level, 1 / 4 * level, 1 - 3 / 4 * level, 1 / 4 * level]
                else:
                    noised_label[i] = [1 / 3 * level, 0, 1 / 3 * level, 1 - 2 / 3 * level]
        elif self.num_classes == 3:
            for i, label in enumerate(train_label):
                if label == 0:
                    noised_label[i] = [1 - 2 / 3 * level, 2 / 3 * level, 0]
                elif label == 1:
                    noised_label[i] = [1 / 3 * level, 1 - 2 / 3 * level, 1 / 3 * level]
                else:
                    noised_label[i] = [0, 2 / 3 * level, 1 - 2 / 3 * level]
        elif self.num_classes == 2:
            for i, label in enumerate(train_label):
                if label == 0:
                    noised_label[i] = [1, 0]
                elif label == 1:
                    noised_label[i] = [0, 1]
        return noised_label
    
    def append(self, edge_index, batch_size):  # stretch and repeat and rename
        edge_index_all = torch.LongTensor(2, edge_index.shape[1] * batch_size)
        data_batch = torch.LongTensor(self.num_nodes * batch_size)
        for i in range((batch_size)):
            edge_index_all[:, i*edge_index.shape[1]:(i+1)*edge_index.shape[1]] = edge_index + i * self.num_nodes
            data_batch[i*self.num_nodes:(i+1)*self.num_nodes] = i
        return edge_index_all.to(edge_index.device), data_batch.to(edge_index.device)
    
    def forward(self, X, alpha=0,need_pred=True,need_dat=False):
        batch_size = len(X.x)
        x, edge_index = X.x, X.edge_index
        x = x.reshape(-1, x.shape[-1])
        edge_index, data_batch = self.append(edge_index, batch_size)
        edge_weight = torch.zeros(
            (self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(
            edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + \
            edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        # edge_index: (2,self.num_nodes*self.num_nodes*batch_size)  edge_weight: (self.num_nodes*self.num_nodes*batch_size,)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # domain classification
        domain_output = None
        if need_dat == True:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
        if need_pred == True:
            x = global_add_pool(x, data_batch, size=batch_size)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc(x)

        # x.shape->(batch_size,num_classes)
        # domain_output.shape->(batch_size*num_nodes,2)

        # NO softmax!!!
        # x=torch.softmax(x,dim=-1)
        # if domain_output is not None:
        #     domain_output=torch.softmax(domain_output,dim=-1)
        if domain_output is not None:
            return  x, domain_output
        else:
            return  x
        # F.log_softmax(outputs, dim=1)
        # return F.log_softmax(x, dim=1)

class SparseL1Regularization(nn.Module):
    def __init__(self, l1_lambda):
        super(SparseL1Regularization, self).__init__()
        self.l1_lambda = l1_lambda

    def forward(self, x):
        l1_norm = torch.norm(x, p=1)
        return self.l1_lambda * l1_norm

def distance_3d_square(a, b):
    return torch.tensor(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2))
    
ele_node_c = [[-21.2, 66.9, 12.1], [1.4, 65.1, 11.3], [24.3, 66.3, 12.5], [-32.7, 48.4, 32.8], [35.1, 50.1, 31.1],
              [-52.1, 28.6, 3.8], [-51.4, 26.7, 24.7], [-39.7, 25.3, 44.7], [-22.1, 26.8, 54.9], [0.0, 26.8, 60.6],
              [23.6, 28.2, 55.6], [41.9, 27.5, 43.9], [52.9, 28.7, 25.2], [53.2, 28.4, 3.1], [-59.2, 3.4, -2.1],
              [-59.1, 3.0, 26.1], [-45.5, 2.4, 51.3], [-24.7, 0.3, 66.4], [1.0, 1.0, 72.8], [26.1, 3.2, 66.0],
              [47.5, 4.6, 49.7], [60.5, 4.9, 25.5], [60.2, 4.7, -2.8], [-65.8, -17.8, -2.9], [-63.6, -18.9, 25.8],
              [-49.1, -20.7, 53.2], [-25.1, -22.5, 70.1], [0.8, -21.9, 77.4], [26.7, -20.9, 69.5], [50.3, -18.8, 53.0],
              [65.2, -18.0, 26.4], [67.4, -18.5, -3.4], [-63.6, -44.7, -4.0], [-61.8, -46.2, 22.5],
              [-46.9, -47.7, 49.7], [-24.0, -49.1, 66.1], [0.7, -47.9, 72.6], [25.8, -47.1, 66.0], [49.5, -45.5, 50.7],
              [62.9, -44.6, 24.4], [64.6, -45.4, -3.7], [-55.9, -64.8, 0.0], [-52.7, -67.1, 19.9], [-41.4, -67.8, 42.4],
              [-21.6, -71.3, 52.6], [0.7, -69.3, 56.9], [24.4, -69.9, 53.5], [44.2, -65.8, 42.7], [54.4, -65.3, 20.2],
              [56.4, -64.4, 0.1], [-44.0, -81.7, 1.6], [-38.5, -83.0, 14.0], [-33.3, -84.3, 26.5], [0.0, -87.9, 33.5],
              [35.2, -82.6, 26.1], [39.3, -82.3, 13.0], [43.3, -82.0, 0.7], [-38.5, -93.3, 5.0], [-25.8, -93.3, 7.7],
              [0.3, -97.1, 8.7], [25.0, -95.2, 6.2], [39.3, -82.3, 5.0]]
SEED_CHANNEL_LIST = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]
SEED_ADJACENCY_LIST = {
    'FP1': ['FPZ', 'AF3'],
    'FPZ': ['FP1', 'FP2'],
    'FP2': ['FPZ', 'AF4'],
    'AF3': ['FP1', 'F5', 'F3', 'F1'],
    'AF4': ['F2', 'F4', 'F6', 'FP2'],
    'F7': ['F5', 'FT7'],
    'F5': ['F7', 'AF3', 'F3', 'FC5'],
    'F3': ['AF3', 'F5', 'FC3', 'F1'],
    'F1': ['AF3', 'F3', 'FC1', 'FZ'],
    'FZ': ['F1', 'FCZ', 'F2'],
    'F2': ['FZ', 'FC2', 'F4', 'AF4'],
    'F4': ['F2', 'FC4', 'F6', 'AF4'],
    'F6': ['AF4', 'F4', 'FC6', 'F8'],
    'F8': ['F6', 'FT8'],
    'FT7': ['F7', 'FC5', 'T7'],
    'FC5': ['F5', 'FT7', 'C5', 'FC3'],
    'FC3': ['F3', 'FC5', 'C3', 'FC1'],
    'FC1': ['F1', 'FC3', 'C1', 'FCZ'],
    'FCZ': ['FZ', 'FC1', 'CZ', 'FC2'],
    'FC2': ['F2', 'FCZ', 'C2', 'FC4'],
    'FC4': ['F4', 'FC2', 'C4', 'FC6'],
    'FC6': ['F6', 'FC4', 'C6', 'FT8'],
    'FT8': ['F8', 'FC6', 'T8'],
    'T7': ['FT7', 'C5', 'TP7'],
    'C5': ['FC5', 'T7', 'C3', 'CP5'],
    'C3': ['FC3', 'C5', 'C1', 'CP3'],
    'C1': ['FC1', 'C3', 'CP1', 'CZ'],
    'CZ': ['FCZ', 'C1', 'CPZ', 'C2'],
    'C2': ['FC2', 'CZ', 'CP2', 'C4'],
    'C4': ['FC4', 'C2', 'CP4', 'C6'],
    'C6': ['FC6', 'C4', 'CP6', 'T8'],
    'T8': ['FT8', 'C6', 'TP8'],
    'TP7': ['T7', 'CP5', 'P7'],
    'CP5': ['C5', 'TP7', 'P5', 'CP3'],
    'CP3': ['C3', 'CP5', 'P3', 'CP1'],
    'CP1': ['C1', 'CP3', 'P1', 'CPZ'],
    'CPZ': ['CZ', 'CP1', 'PZ', 'CP2'],
    'CP2': ['C2', 'CPZ', 'P2', 'CP4'],
    'CP4': ['C4', 'CP2', 'P4', 'CP6'],
    'CP6': ['C6', 'CP4', 'P6', 'TP8'],
    'TP8': ['T8', 'CP6', 'P8'],
    'P7': ['TP7', 'P5', 'PO7'],
    'P5': ['CP5', 'P7', 'PO5', 'P3'],
    'P3': ['CP3', 'P5', 'P1'],
    'P1': ['CP1', 'P3', 'PO3', 'PZ'],
    'PZ': ['CPZ', 'P1', 'POZ', 'P2'],
    'P2': ['CP2', 'PZ', 'PO4', 'P4'],
    'P4': ['CP4', 'P2', 'P6'],
    'P6': ['CP6', 'P4', 'P8'],
    'P8': ['TP8', 'P6', 'PO8'],
    'PO7': ['P7', 'PO5', 'CB1'],
    'PO5': ['P5', 'PO7', 'CB1', 'PO3'],
    'PO3': ['P1', 'PO5', 'O1', 'POZ'],
    'POZ': ['PZ', 'PO3', 'OZ', 'PO4'],
    'PO4': ['P2', 'POZ', 'O2', 'PO6'],
    'PO6': ['P6', 'PO4', 'CB2', 'PO8'],
    'PO8': ['P8', 'PO6', 'CB2'],
    'CB1': ['PO7', 'PO5', 'O1'],
    'O1': ['CB1', 'PO3', 'OZ'],
    'OZ': ['POZ', 'O1', 'O2'],
    'O2': ['PO4', 'OZ', 'CB2'],
    'CB2': ['PO6', 'O2', 'PO8']
}
DEFAULT_GLOBAL_CHANNEL_LIST = [('FP1', 'FP2'), ('AF3', 'AF4'), ('F5', 'F6'),
                               ('FC5', 'FC6'), ('C5', 'C6'), ('CP5', 'CP6'),
                               ('P5', 'P6'), ('PO5', 'PO6'), ('O1', 'O2')]
def format_adj_matrix_from_standard(
    channel_list: List,
    standard_channel_location_dict: Dict,
    delta: float = 0.00056,
    global_channel_list: List[Tuple[str]] = DEFAULT_GLOBAL_CHANNEL_LIST
) -> List[List]:
    r'''
    Creates an adjacency matrix based on the relative positions of electrodes in a standard system, allowing the addition of global electrode links to connect non-adjacent but symmetrical electrodes.

    - Paper: Zhong P, Wang D, Miao C. EEG-based emotion recognition using regularized graph neural networks[J]. IEEE Transactions on Affective Computing, 2020.
    - URL: https://ieeexplore.ieee.org/abstract/document/9091308
    - Related Project: https://github.com/zhongpeixiang/RGNN

    Args:
        channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
        standard_channel_location_dict (dict): Electrode location information. Represented in dictionary form, where :obj:`key` corresponds to the electrode name and :obj:`value` corresponds to (x, y, z) of the electrode on the grid. please refer to STANDARD_1005_CHANNEL_LOCATION_DICT and STANDARD_1020_CHANNEL_LOCATION_DICT.
        delta (float): The calibration constant. Due to differences in electrode coordinate scales, the values in the original paper are not recommended. 0.00056 means 20% more nodes are connected to each other. (default: :obj:`0.00056`)
        global_channel_list (float): To leverage the differential asymmetry information, the authors initialize the global inter-channel relations in the adjacency matrix. (default: :obj:`[('FP1', 'FP2'), ('AF3', 'AF4'), ('F5', 'F6'), ('FC5', 'FC6'), ('C5', 'C6'), ('CP5', 'CP6'), ('P5', 'P6'), ('PO5', 'PO6'), ('O1', 'O2')]`)
    '''
    node_map = {k: i for i, k in enumerate(channel_list)}
    adj_matrix = np.zeros((len(channel_list), len(channel_list)))

    for start_node_name in channel_list:
        if not start_node_name in standard_channel_location_dict:
            continue
        for end_node_name in channel_list:
            if not end_node_name in standard_channel_location_dict:
                continue
            start_node_pos = np.array(
                standard_channel_location_dict[start_node_name])
            end_node_pos = np.array(
                standard_channel_location_dict[end_node_name])
            edge_weight = np.linalg.norm(start_node_pos - end_node_pos)
            edge_weight = min(1.0, delta / (edge_weight**2 + 1e-6))

            adj_matrix[node_map[start_node_name]][
                node_map[end_node_name]] = edge_weight

    for start_node_name, end_node_name in global_channel_list:
        if (not start_node_name in node_map) or (not end_node_name in node_map):
            continue
        adj_matrix[node_map[start_node_name]][
            node_map[end_node_name]] = adj_matrix[node_map[start_node_name]][
                node_map[end_node_name]] - 1.0

    return adj_matrix.tolist()
STANDARD_1005_CHANNEL_LOCATION_DICT = {
    'FP1': [-0.0294367, 0.08391710000000001, -0.0069900000000000006],
    'FPZ': [0.00011229999999999999, 0.088247, -0.0017130000000000001],
    'FP2': [0.029872299999999997, 0.0848959, -0.00708],
    'AF9': [-0.048970799999999995, 0.0640872, -0.047683],
    'AF7': [-0.0548397, 0.0685722, -0.01059],
    'AF5': [-0.045430700000000004, 0.0728622, 0.005978],
    'AF3': [-0.0337007, 0.0768371, 0.021227],
    'AF1': [-0.018471699999999997, 0.0799041, 0.032752],
    'AFZ': [0.0002313, 0.080771, 0.035417000000000004],
    'AF2': [0.0198203, 0.08030190000000001, 0.032764],
    'AF4': [0.0357123, 0.0777259, 0.021956],
    'AF6': [0.0465843, 0.0738078, 0.006033999999999999],
    'AF8': [0.055743299999999996, 0.0696568, -0.010755],
    'AF10': [0.0504352, 0.0638698, -0.048005000000000006],
    'F9': [-0.0701019, 0.041652299999999996, -0.049951999999999996],
    'F7': [-0.0702629, 0.0424743, -0.01142],
    'F5': [-0.0644658, 0.048035299999999996, 0.016921],
    'F3': [-0.0502438, 0.0531112, 0.042192],
    'F1': [-0.0274958, 0.0569311, 0.060342],
    'FZ': [0.0003122, 0.058512, 0.06646200000000001],
    'F2': [0.029514199999999997, 0.0576019, 0.059539999999999996],
    'F4': [0.0518362, 0.0543048, 0.040814],
    'F6': [0.0679142, 0.049829700000000005, 0.016367],
    'F8': [0.0730431, 0.0444217, -0.012],
    'F10': [0.07211409999999999, 0.0420667, -0.050452],
    'FT9': [-0.08407590000000001, 0.0145673, -0.050429],
    'FT7': [-0.080775, 0.0141203, -0.011134999999999999],
    'FC5': [-0.0772149, 0.0186433, 0.024460000000000003],
    'FC3': [-0.060181899999999997, 0.0227162, 0.055543999999999996],
    'FC1': [-0.0340619, 0.0260111, 0.07998699999999999],
    'FCZ': [0.0003761, 0.02739, 0.08866800000000001],
    'FC2': [0.034784100000000005, 0.0264379, 0.078808],
    'FC4': [0.062293100000000004, 0.0237228, 0.055630000000000006],
    'FC6': [0.0795341, 0.0199357, 0.024437999999999998],
    'FT8': [0.0818151, 0.0154167, -0.01133],
    'FT10': [0.0841131, 0.0143647, -0.050538],
    'T9': [-0.0858941, -0.0158287, -0.048283],
    'T7': [-0.0841611, -0.0160187, -0.009346],
    'C5': [-0.08028010000000001, -0.0137597, 0.02916],
    'C3': [-0.06535809999999999, -0.0116317, 0.064358],
    'C1': [-0.036158, -0.0099839, 0.089752],
    'CZ': [0.0004009, -0.009167, 0.100244],
    'C2': [0.037672, -0.0096241, 0.088412],
    'C4': [0.06711790000000001, -0.0109003, 0.06358],
    'C6': [0.0834559, -0.0127763, 0.029207999999999998],
    'T8': [0.0850799, -0.0150203, -0.00949],
    'T10': [0.0855599, -0.0163613, -0.048271],
    'TP9': [-0.0856192, -0.0465147, -0.045707],
    'TP7': [-0.08483020000000001, -0.046021700000000006, -0.007056],
    'CP5': [-0.0795922, -0.0465507, 0.030949],
    'CP3': [-0.0635562, -0.0470088, 0.065624],
    'CP1': [-0.0355131, -0.0472919, 0.091315],
    'CPZ': [0.0003858, -0.047318, 0.099432],
    'CP2': [0.0383838, -0.0470731, 0.090695],
    'CP4': [0.0666118, -0.0466372, 0.06558],
    'CP6': [0.0833218, -0.046101300000000005, 0.031206],
    'TP8': [0.0855488, -0.0455453, -0.00713],
    'TP10': [0.0861618, -0.0470353, -0.045869],
    'P9': [-0.0730093, -0.07376569999999999, -0.040998],
    'P7': [-0.0724343, -0.0734527, -0.002487],
    'P5': [-0.06727230000000001, -0.0762907, 0.028382],
    'P3': [-0.0530073, -0.0787878, 0.05594],
    'P1': [-0.0286203, -0.0805249, 0.075436],
    'PZ': [0.0003247, -0.08111499999999999, 0.082615],
    'P2': [0.0319197, -0.08048709999999999, 0.07671599999999999],
    'P4': [0.0556667, -0.0785602, 0.056561],
    'P6': [0.0678877, -0.07590430000000001, 0.028091],
    'P8': [0.0730557, -0.07306829999999999, -0.00254],
    'P10': [0.0738947, -0.07439029999999999, -0.04122],
    'PO9': [-0.054910400000000005, -0.0980448, -0.035465],
    'PO7': [-0.054840400000000004, -0.0975279, 0.0027919999999999998],
    'PO5': [-0.0484244, -0.0993408, 0.021599],
    'PO3': [-0.0365114, -0.10085290000000001, 0.037167],
    'PO1': [-0.0189724, -0.101768, 0.046536],
    'POZ': [0.0002156, -0.10217799999999999, 0.050608],
    'PO2': [0.019877600000000002, -0.10179300000000001, 0.046393000000000004],
    'PO4': [0.0367816, -0.10084910000000001, 0.036397],
    'PO6': [0.0498196, -0.0994461, 0.021727],
    'PO8': [0.055666600000000004, -0.0976251, 0.00273],
    'PO10': [0.0549876, -0.0980911, -0.035540999999999996],
    'O1': [-0.0294134, -0.112449, 0.008839],
    'OZ': [0.0001076, -0.114892, 0.014657],
    'O2': [0.0298426, -0.112156, 0.0088],
    'I1': [-0.029818400000000002, -0.11456999999999999, -0.029216000000000002],
    'IZ': [4.499999999999999e-06, -0.118565, -0.023077999999999998],
    'I2': [0.0297416, -0.11426, -0.029256],
    'AFP9H': [-0.0432897, 0.0758552, -0.028244],
    'AFP7H': [-0.038551699999999994, 0.0799532, -0.004995],
    'AFP5H': [-0.027985700000000002, 0.08245910000000001, 0.002702],
    'AFP3H': [-0.0171947, 0.08484910000000001, 0.010027],
    'AFP1H': [-0.0059317, 0.086878, 0.0162],
    'AFP2H': [0.0071053, 0.087074, 0.016469],
    'AFP4H': [0.0189233, 0.0855969, 0.011443],
    'AFP6H': [0.0286443, 0.08297589999999999, 0.002828],
    'AFP8H': [0.0393203, 0.0806868, -0.004725],
    'AFP10H': [0.0438223, 0.0765418, -0.028307],
    'AFF9H': [-0.0632538, 0.053857300000000004, -0.030316],
    'AFF7H': [-0.0613508, 0.058799199999999996, 0.000897],
    'AFF5H': [-0.0507998, 0.0640412, 0.023089],
    'AFF3H': [-0.0343157, 0.0683931, 0.041188],
    'AFF1H': [-0.0114357, 0.0707561, 0.050348],
    'AFF2H': [0.0134793, 0.07120099999999999, 0.051175],
    'AFF4H': [0.0361833, 0.06915089999999999, 0.041254],
    'AFF6H': [0.0523972, 0.06507080000000001, 0.022861999999999997],
    'AFF8H': [0.0629152, 0.0600448, 0.00063],
    'AFF10H': [0.0643342, 0.054599800000000004, -0.030444],
    'FFT9H': [-0.07906690000000001, 0.0280813, -0.031253],
    'FFT7H': [-0.0744999, 0.0313003, 0.0048460000000000005],
    'FFC5H': [-0.0652379, 0.036428199999999994, 0.036143999999999996],
    'FFC3H': [-0.0444098, 0.0407622, 0.061689999999999995],
    'FFC1H': [-0.0154238, 0.04366, 0.077682],
    'FFC2H': [0.0175922, 0.044054, 0.077788],
    'FFC4H': [0.045853200000000004, 0.0416228, 0.060647],
    'FFC6H': [0.06712810000000001, 0.037799799999999995, 0.035296],
    'FFT8H': [0.0780531, 0.032981699999999996, 0.004483],
    'FFT10H': [0.0800971, 0.0285137, -0.031338],
    'FTT9H': [-0.084125, -0.0018467, -0.029794],
    'FTT7H': [-0.082355, 0.0008263000000000001, 0.008579],
    'FCC5H': [-0.074692, 0.0043033, 0.045307],
    'FCC3H': [-0.051050899999999996, 0.0071772, 0.074377],
    'FCC1H': [-0.018219000000000003, 0.009094099999999999, 0.092529],
    'FCC2H': [0.018786999999999998, 0.0092479, 0.091562],
    'FCC4H': [0.0518851, 0.0077978, 0.073507],
    'FCC6H': [0.077002, 0.0053357000000000005, 0.04535],
    'FTT8H': [0.083888, 0.0019457, 0.008501],
    'FTT10H': [0.084123, -0.0018083, -0.029638],
    'TTP9H': [-0.0869731, -0.0322157, -0.027847999999999998],
    'TTP7H': [-0.0855651, -0.0306287, 0.011153],
    'CCP5H': [-0.0764071, -0.0297307, 0.049217],
    'CCP3H': [-0.0529281, -0.0289058, 0.080304],
    'CCP1H': [-0.018354099999999998, -0.0283219, 0.09822],
    'CCP2H': [0.0202199, -0.0281481, 0.098172],
    'CCP4H': [0.0551139, -0.0283862, 0.080474],
    'CCP6H': [0.07900589999999999, -0.0289863, 0.049628],
    'TTP8H': [0.08599989999999999, -0.0298203, 0.011248],
    'TTP10H': [0.08862489999999999, -0.032272300000000004, -0.028],
    'TPP9H': [-0.0781602, -0.060756700000000004, -0.023824],
    'TPP7H': [-0.0766802, -0.060831699999999995, 0.01288],
    'CPP5H': [-0.0681152, -0.0629747, 0.047252],
    'CPP3H': [-0.0469142, -0.06469079999999999, 0.075296],
    'CPP1H': [-0.0158202, -0.0655999, 0.091164],
    'CPP2H': [0.019419799999999997, -0.065595, 0.092405],
    'CPP4H': [0.0506738, -0.0644822, 0.07612999999999999],
    'CPP6H': [0.0710958, -0.0626243, 0.047328],
    'TPP8H': [0.0785198, -0.0604323, 0.012901999999999999],
    'TPP10H': [0.07890269999999999, -0.060955300000000004, -0.023805],
    'PPO9H': [-0.06459730000000001, -0.0876558, -0.019014],
    'PPO7H': [-0.0629593, -0.08750279999999999, 0.012952],
    'PPO5H': [-0.054010300000000004, -0.0898988, 0.037332000000000004],
    'PPO3H': [-0.0358874, -0.0916669, 0.055504],
    'PPO1H': [-0.0120474, -0.09260689999999999, 0.065508],
    'PPO2H': [0.013922599999999999, -0.092694, 0.066958],
    'PPO4H': [0.0377986, -0.09162909999999999, 0.056733],
    'PPO6H': [0.054608699999999996, -0.08964019999999999, 0.037035],
    'PPO8H': [0.06311169999999999, -0.0872282, 0.012856],
    'PPO10H': [0.0650137, -0.0878062, -0.018952],
    'POO9H': [-0.0428624, -0.10807299999999999, -0.013151],
    'POO7H': [-0.040120399999999994, -0.107129, 0.012061],
    'POO5H': [-0.0319514, -0.108252, 0.023047],
    'POO3H': [-0.019862400000000002, -0.108942, 0.02976],
    'POO1H': [-0.0069194, -0.10926000000000001, 0.03271],
    'POO2H': [0.0068036, -0.109163, 0.031582],
    'POO4H': [0.020293600000000002, -0.108914, 0.028943999999999998],
    'POO6H': [0.032175600000000006, -0.108252, 0.022255],
    'POO8H': [0.0410976, -0.10724500000000001, 0.012138],
    'POO10H': [0.0438946, -0.109127, -0.01317],
    'OI1H': [-0.0148504, -0.117987, -0.00692],
    'OI2H': [0.0150946, -0.118018, -0.006933],
    'FP1H': [-0.014810700000000001, 0.0872351, -0.004477],
    'FP2H': [0.0151623, 0.08809099999999999, -0.004551],
    'AF9H': [-0.0548298, 0.0664132, -0.029704],
    'AF7H': [-0.0511757, 0.0708362, -0.0017549999999999998],
    'AF5H': [-0.0396407, 0.07486709999999999, 0.013678000000000001],
    'AF3H': [-0.0272187, 0.0787091, 0.028375],
    'AF1H': [-0.0091977, 0.0806051, 0.035133000000000005],
    'AF2H': [0.0104823, 0.08086499999999999, 0.035359],
    'AF4H': [0.0285803, 0.0793029, 0.02847],
    'AF6H': [0.0409403, 0.0757399, 0.013859999999999999],
    'AF8H': [0.0520293, 0.0718468, -0.0019199999999999998],
    'AF10H': [0.0557542, 0.0671698, -0.029824000000000003],
    'F9H': [-0.07150790000000001, 0.041119300000000004, -0.030854],
    'F7H': [-0.0685558, 0.0452843, 0.003002],
    'F5H': [-0.0584878, 0.050672199999999994, 0.030192],
    'F3H': [-0.039979799999999996, 0.0552601, 0.0526],
    'F1H': [-0.013383800000000001, 0.0579021, 0.064332],
    'F2H': [0.0158342, 0.0584559, 0.06499200000000001],
    'F4H': [0.0417942, 0.0562259, 0.051499],
    'F6H': [0.0600522, 0.0520858, 0.028707999999999997],
    'F8H': [0.0719592, 0.047191699999999996, 0.002475],
    'F10H': [0.0727981, 0.041821800000000006, -0.031026],
    'FT9H': [-0.0829559, 0.0133203, -0.030808],
    'FT7H': [-0.0801139, 0.0163903, 0.006849999999999999],
    'FC5H': [-0.0712099, 0.0208203, 0.041324],
    'FC3H': [-0.0485119, 0.0245292, 0.06913599999999999],
    'FC1H': [-0.017343900000000002, 0.027024100000000002, 0.086923],
    'FC2H': [0.0184181, 0.0272709, 0.086437],
    'FC4H': [0.0495481, 0.0252378, 0.06843],
    'FC6H': [0.0732191, 0.022006699999999997, 0.041297],
    'FT8H': [0.0815801, 0.0176837, 0.006564],
    'FT10H': [0.0833711, 0.013547700000000001, -0.030749],
    'T9H': [-0.08513209999999999, -0.0170557, -0.028731000000000003],
    'T7H': [-0.0829461, -0.0148827, 0.010009],
    'C5H': [-0.0752941, -0.0126397, 0.047904],
    'C3H': [-0.0515811, -0.0107548, 0.078035],
    'C1H': [-0.018279, -0.0094319, 0.097356],
    'C2H': [0.019678, -0.0093041, 0.095706],
    'C4H': [0.053805900000000004, -0.010144199999999999, 0.07773000000000001],
    'C6H': [0.0781249, -0.0117353, 0.04784],
    'T8H': [0.0851369, -0.0139063, 0.009890000000000001],
    'T10H': [0.08609990000000001, -0.0170883, -0.028756],
    'TP9H': [-0.08481019999999999, -0.0472457, -0.02622],
    'TP7H': [-0.0827042, -0.0462977, 0.011974],
    'CP5H': [-0.0733012, -0.0467917, 0.049109],
    'CP3H': [-0.051049199999999996, -0.047175800000000004, 0.080016],
    'CP1H': [-0.0173542, -0.0473419, 0.09741],
    'CP2H': [0.0206798, -0.047232100000000006, 0.098072],
    'CP4H': [0.0539968, -0.0468902, 0.080077],
    'CP6H': [0.0765498, -0.0463733, 0.04914],
    'TP8H': [0.08519979999999999, -0.045807299999999995, 0.012102],
    'TP10H': [0.0854428, -0.0472213, -0.026175999999999998],
    'P9H': [-0.0721773, -0.0746277, -0.021536000000000003],
    'P7H': [-0.07011329999999999, -0.0748677, 0.012999],
    'P5H': [-0.0617283, -0.0776238, 0.043028],
    'P3H': [-0.041673299999999996, -0.0797528, 0.066715],
    'P1H': [-0.0139613, -0.0810029, 0.081003],
    'P2H': [0.0172977, -0.080981, 0.081641],
    'P4H': [0.0447477, -0.07961109999999999, 0.067655],
    'P6H': [0.0636267, -0.0773022, 0.043119],
    'P8H': [0.0721037, -0.0744993, 0.013025],
    'P10H': [0.0732817, -0.0750773, -0.021576],
    'PO9H': [-0.054775399999999995, -0.0989768, -0.016193000000000003],
    'PO7H': [-0.051928400000000006, -0.0984438, 0.012304],
    'PO5H': [-0.043342399999999996, -0.1001629, 0.030009],
    'PO3H': [-0.028007400000000002, -0.101361, 0.042379],
    'PO1H': [-0.009503399999999999, -0.10206, 0.049418],
    'PO2H': [0.0102356, -0.102029, 0.048942],
    'PO4H': [0.028647600000000002, -0.10139010000000001, 0.042137999999999995],
    'PO6H': [0.0442206, -0.10021909999999999, 0.029808],
    'PO8H': [0.0528386, -0.098536, 0.01225],
    'PO10H': [0.0558596, -0.09989400000000001, -0.016207999999999997],
    'O1H': [-0.0148054, -0.1151, 0.011829000000000001],
    'O2H': [0.0151456, -0.115191, 0.011833],
    'I1H': [-0.0151584, -0.118242, -0.026047999999999998],
    'I2H': [0.0151286, -0.11815099999999999, -0.026081],
    'AFP9': [-0.036124699999999996, 0.0723801, -0.045852],
    'AFP7': [-0.0435117, 0.0785802, -0.00924],
    'AFP5': [-0.0332847, 0.08120709999999999, -0.00114],
    'AFP3': [-0.022351700000000002, 0.0835621, 0.006071],
    'AFP1': [-0.0122417, 0.08619410000000001, 0.014188000000000001],
    'AFPZ': [0.00017030000000000002, 0.087322, 0.017442],
    'AFP2': [0.013622299999999999, 0.08675790000000001, 0.015302],
    'AFP4': [0.0241013, 0.0843769, 0.0074329999999999995],
    'AFP6': [0.0339133, 0.08181189999999999, -0.001035],
    'AFP8': [0.0439483, 0.0792958, -0.009300000000000001],
    'AFP10': [0.0377123, 0.07216790000000001, -0.046197],
    'AFF9': [-0.0593398, 0.052680199999999996, -0.04877],
    'AFF7': [-0.0632618, 0.0559922, -0.011173],
    'AFF5': [-0.0558198, 0.0613962, 0.011884],
    'AFF3': [-0.0433817, 0.0663672, 0.032811],
    'AFF1': [-0.0235817, 0.06991710000000001, 0.047293],
    'AFFZ': [0.0002763, 0.07128, 0.052092],
    'AFF2': [0.0255583, 0.07055589999999999, 0.047827],
    'AFF4': [0.0451522, 0.0672748, 0.032731],
    'AFF6': [0.0580002, 0.0625998, 0.0119],
    'AFF8': [0.0646732, 0.0572738, -0.011460000000000001],
    'AFF10': [0.0606012, 0.0522668, -0.049038],
    'FFT9': [-0.07848390000000001, 0.0287703, -0.050522],
    'FFT7': [-0.0766149, 0.028653300000000003, -0.011508],
    'FFC5': [-0.0715059, 0.0339263, 0.020992999999999998],
    'FFC3': [-0.0559399, 0.0387162, 0.049788],
    'FFC1': [-0.030654800000000003, 0.042415100000000004, 0.07104],
    'FFCZ': [0.0003512, 0.044073999999999995, 0.079141],
    'FFC2': [0.032645099999999996, 0.043100900000000004, 0.070795],
    'FFC4': [0.0575042, 0.0398518, 0.048811],
    'FFC6': [0.0742501, 0.035499699999999995, 0.02038],
    'FFT8': [0.0790341, 0.030343699999999998, -0.011997],
    'FFT10': [0.07992010000000001, 0.0289417, -0.050914],
    'FTT9': [-0.087362, -0.0005147000000000001, -0.049837000000000006],
    'FTT7': [-0.082668, -0.0009417, -0.010284000000000001],
    'FCC5': [-0.080133, 0.0025853, 0.027312],
    'FCC3': [-0.064161, 0.005831299999999999, 0.060884999999999995],
    'FCC1': [-0.035749, 0.008309100000000002, 0.08545900000000001],
    'FCCZ': [0.0003911, 0.009507999999999999, 0.09556],
    'FCC2': [0.03607, 0.008651899999999999, 0.08383199999999999],
    'FCC4': [0.065164, 0.0066197999999999995, 0.060052],
    'FCC6': [0.08154399999999999, 0.0036636999999999998, 0.027201],
    'FTT8': [0.083168, 0.0001817, -0.010364],
    'FTT10': [0.085393, -0.0009523, -0.04952],
    'TTP9': [-0.08663209999999999, -0.0312377, -0.047178],
    'TTP7': [-0.0859331, -0.0310927, -0.008474],
    'CCP5': [-0.0815431, -0.0301727, 0.030273],
    'CCP3': [-0.06612810000000001, -0.0292957, 0.065898],
    'CCP1': [-0.0369301, -0.028569900000000002, 0.091734],
    'CCPZ': [0.0003959, -0.028163, 0.10126900000000001],
    'CCP2': [0.0385399, -0.0282251, 0.090976],
    'CCP4': [0.0688539, -0.0286403, 0.06641],
    'CCP6': [0.0845529, -0.0293783, 0.030878],
    'TTP8': [0.08599989999999999, -0.0302803, -0.008435],
    'TTP10': [0.0867619, -0.031731300000000004, -0.047253],
    'TPP9': [-0.0807152, -0.0606457, -0.043594],
    'TPP7': [-0.0785992, -0.0597237, -0.004758],
    'CPP5': [-0.0736642, -0.0619227, 0.030379999999999997],
    'CPP3': [-0.059411200000000004, -0.0639248, 0.06267199999999999],
    'CPP1': [-0.032728299999999995, -0.0653199, 0.085944],
    'CPPZ': [0.0003658, -0.06575, 0.094058],
    'CPP2': [0.0358918, -0.06513809999999999, 0.08598],
    'CPP4': [0.0622558, -0.0636152, 0.062719],
    'CPP6': [0.0766708, -0.0615483, 0.030543],
    'TPP8': [0.0793188, -0.0593033, -0.00484],
    'TPP10': [0.0815598, -0.0612153, -0.0438],
    'PPO9': [-0.0645703, -0.08643179999999999, -0.038324],
    'PPO7': [-0.0645833, -0.0862218, 3.3e-05],
    'PPO5': [-0.0587123, -0.0887048, 0.025193],
    'PPO3': [-0.0461603, -0.0908878, 0.047445999999999995],
    'PPO1': [-0.024648299999999998, -0.0922919, 0.062076],
    'PPOZ': [0.0002727, -0.092758, 0.067342],
    'PPO2': [0.026436699999999997, -0.0922951, 0.06319899999999999],
    'PPO4': [0.047143700000000004, -0.09071219999999999, 0.047678],
    'PPO6': [0.0608127, -0.08850419999999999, 0.025661999999999997],
    'PPO8': [0.0651517, -0.08594320000000001, -8.999999999999999e-06],
    'PPO10': [0.0650377, -0.0867182, -0.038448],
    'POO9': [-0.0431284, -0.107516, -0.032387],
    'CB1': [-0.0429764, -0.10649299999999999, 0.0057729999999999995],
    'POO5': [-0.0362344, -0.10771599999999999, 0.01775],
    'POO3': [-0.0259844, -0.108616, 0.026544],
    'POO1': [-0.0136644, -0.109266, 0.032856],
    'POOZ': [0.0001676, -0.109276, 0.03279],
    'POO2': [0.0136506, -0.109106, 0.030935999999999998],
    'POO4': [0.0266636, -0.108668, 0.026414999999999998],
    'POO6': [0.0377006, -0.10784, 0.018068999999999998],
    'CB2': [0.0436696, -0.106599, 0.005726],
    'POO10': [0.0431766, -0.107444, -0.032463],
    'OI1': [-0.0293914, -0.114511, -0.01002],
    'OIZ': [5.2499999999999995e-05, -0.119343, -0.003936],
    'OI2': [0.029552600000000002, -0.113636, -0.010051000000000001],
    'T3': [-0.0841611, -0.0160187, -0.009346],
    'T5': [-0.0724343, -0.0734527, -0.002487],
    'T4': [0.0850799, -0.0150203, -0.00949],
    'T6': [0.0730557, -0.07306829999999999, -0.00254],
    'M1': [-0.0860761, -0.0449897, -0.067986],
    'M2': [0.08579389999999999, -0.0450093, -0.06803100000000001],
    'A1': [-0.0860761, -0.0249897, -0.067986],
    'A2': [0.08579389999999999, -0.025009299999999998, -0.06803100000000001]
}

SEED_ADJACENCY_MATRIX = format_adj_matrix_from_standard(SEED_CHANNEL_LIST,
                                                        STANDARD_1005_CHANNEL_LOCATION_DICT)
