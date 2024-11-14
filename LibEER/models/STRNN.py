import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim

from tqdm import tqdm
import yaml

from utils.store import save_state
from utils.metric import Metric

param_path = 'config/model_param/STRNN.yaml'


class STRNN(nn.Module):
    def __init__(self, sample_length=9, num_electrodes=62, in_channels=5, num_classes=3, sp_hidden=30, tp_hidden=30
                 , sp_projection=10, tp_projection=5):
        super(STRNN, self).__init__()
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sp_hidden = sp_hidden
        self.tp_hidden = tp_hidden
        self.sp_projection = sp_projection
        self.tp_projection = tp_projection
        self.sample_length = sample_length
        self.get_param()
        self.directions = []
        self.pos = []
        if self.num_electrodes == 62:
            self.directions = Sixtytwo_channel_directions
            self.pos = Sixtytwo_channel_coor

        self.sp_rnn = SRNN(num_electrodes=self.num_electrodes, in_channels=self.in_channels, num_hidden=self.sp_hidden,
                           num_projection=self.sp_projection, directions=self.directions, pos=self.pos)
        self.f_tp_rnn = TRNN(sample_length=self.sample_length, num_hidden=self.tp_hidden,
                             num_projection=self.tp_projection)
        self.b_tp_rnn = TRNN(sample_length=self.sample_length, num_hidden=self.tp_hidden,
                             num_projection=self.tp_projection)
        self.fp_1 = nn.Linear(self.tp_hidden, 1)
        self.fp_2 = nn.Linear(self.tp_projection, self.num_classes)

        self.bp_1 = nn.Linear(self.tp_hidden, 1)
        self.bp_2 = nn.Linear(self.tp_projection, self.num_classes)

    def get_param(self):
        return

    def init_weight(self):
        nn.init.xavier_normal_(self.fp_1.weight)
        nn.init.zeros_(self.fp_1.bias)
        nn.init.xavier_normal_(self.fp_2.weight)
        nn.init.zeros_(self.fp_2.bias)
        nn.init.xavier_normal_(self.bp_1.weight)
        nn.init.zeros_(self.bp_1.bias)
        nn.init.xavier_normal_(self.bp_2.weight)
        nn.init.zeros_(self.bp_2.bias)

    def forward(self, x):
        # shape of x -> (batch_size, sample_length, num_electrodes, num_features)
        # reshape to -> (sample_length, batch_size, num_electrodes, num_features)
        x = x.permute(1, 0, 2, 3)
        ms = None
        for idx, x_i in enumerate(x):
            if idx == 0:
                ms = self.sp_rnn(x_i).unsqueeze(dim=0)
            else:
                ms = torch.concatenate((ms, self.sp_rnn(x_i).unsqueeze(dim=0)), dim=0)
        # ms shape: (t, batch_size, sp_hidden)
        q_f = self.f_tp_rnn(ms)
        q_b = self.b_tp_rnn(torch.flip(ms, dims=[0]))
        # concatenated the forward and backward temporal output
        output = self.fp_2(torch.squeeze(self.fp_1(q_f), dim=-1)) + self.bp_2(torch.squeeze(self.bp_1(q_b), dim=-1))
        return output


class SRNN(nn.Module):
    def __init__(self, num_electrodes, in_channels, num_hidden, num_projection, directions, pos):
        super(SRNN, self).__init__()
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.num_hidden = num_hidden
        self.num_projection = num_projection
        self.pos = pos
        self.directions = directions

        # self.hiddens = nn.ModuleList()
        # # There are hidden layers for each channel in each direction
        # for _ in range(len(directions)):
        #     hidden = nn.ModuleList()
        #     # for _ in range(len(num_electrodes)):
        #     #     hidden.append(nn.Parameter(torch.tensor(self.num_hidden), requires_grad=True))
        #     self.hiddens.append(hidden)

        self.Ns = nn.Parameter(torch.zeros((len(self.directions), self.num_electrodes, self.num_electrodes)),
                               requires_grad=False)
        for di, direction in enumerate(self.directions):
            for pi, point in enumerate(direction):
                for ni in n_set(pi, di, self.pos, direction):
                    self.Ns[di][pi][ni] = 1

        # Process input x using matrix U
        self.sp_Ums = nn.ModuleList()
        for _ in directions:
            self.sp_Ums.append(nn.Linear(self.in_channels, self.num_hidden, bias=True))

        # Update the hidden layer with matrix W
        self.sp_Wms = nn.ModuleList()
        for _ in directions:
            self.sp_Wms.append(nn.Linear(self.num_hidden, self.num_hidden, bias=True))

        # the bias in SRNN
        self.sp_bs = nn.Parameter(torch.Tensor(len(directions), self.num_hidden), requires_grad=True)
        self.relu = nn.ReLU()

        # the projection matrx to downsample
        self.sp_projections = nn.ModuleList()
        for _ in directions:
            self.sp_projections.append(nn.Linear(self.num_electrodes, self.num_projection))

        # gather each direction feature
        self.sp_ps = nn.Linear(self.num_projection * len(directions), 1)
        self.init_weight()

    def init_weight(self):
        for i in range(len(self.directions)):
            nn.init.xavier_normal_(self.sp_Ums[i].weight)
            nn.init.zeros_(self.sp_Ums[i].bias)
            nn.init.xavier_normal_(self.sp_Wms[i].weight)
            nn.init.zeros_(self.sp_Wms[i].bias)
            nn.init.zeros_(self.sp_bs)
            nn.init.xavier_normal_(self.sp_projections[i].weight)
            nn.init.zeros_(self.sp_projections[i].bias)
        nn.init.xavier_normal_(self.sp_ps.weight)
        nn.init.zeros_(self.sp_ps.bias)

    def forward(self, x):
        # x shape->(batch size, num_ele, num_feature)
        s = None
        # # hiddens shape -> (num_directions, num_ele, batch_size, num_hidden)
        # hiddens = torch.zeros((len(self.directions), x.shape[1], x.shape[0], self.num_hidden), device=x.device)

        for ri, direction in enumerate(self.directions):
            # update all the hidden layers in one direction
            hidden_di = torch.zeros((x.shape[0], 1, self.num_hidden), device=x.device)
            for pi, point in enumerate(direction):
                # Iterate through the hidden layer in a specific direction

                # find the neighbors near
                # neighbor_indexes = n_set(pi, ri, self.pos, direction)
                # hidden layer shape of one electrode -> (batch, num_hidden)
                # print(self.Ns[ri][pi][0:pi+1].unsqueeze(0).repeat(x.shape[0], 1, pi+1).shape)
                if pi == 0:
                    h_aggregation = torch.zeros((x.shape[0], self.num_hidden), device=x.device)
                else:
                    h_aggregation = torch.matmul(self.Ns[ri][pi][0:pi].unsqueeze(0).repeat(x.shape[0], 1, 1),
                                                 hidden_di).squeeze(1)
                # Adjust the hidden layer based on the input
                hidden_pi = self.sp_Ums[ri](x[:, point - 1]) + self.sp_Wms[ri](h_aggregation) + self.sp_bs[ri]
                hidden_pi = self.relu(hidden_pi)
                # # hiddens[ri][pi] += self.sp_Ums[ri](x[:, point - 1])
                # for ni in neighbor_indexes:
                #     # Adjust hidden layers based on past ones
                #     hidden_pi += self.sp_Wms[ri](hidden_di[ni])
                # the bias add on the hidden layers
                # hidden_pi += self.sp_bs[ri]

                if pi == 0:
                    hidden_di = hidden_pi.unsqueeze(dim=1)
                else:
                    hidden_di = torch.concatenate((hidden_di, hidden_pi.unsqueeze(dim=1)), dim=1)
            # hidden_di shape -> (batch_size, num_ele, num_hidden)
            # reshape hidden_di to -> (batch_size, num_hidden, num_ele)
            # project the all electrodes to the output feature vector
            if ri == 0:
                s = self.sp_projections[ri](hidden_di.permute(0, 2, 1))
            else:
                s = torch.concatenate((s, self.sp_projections[ri](hidden_di.permute(0, 2, 1))), dim=-1)
        # s shape -> (batch_size, num_hidden, num_pro * len(direction))
        m = torch.squeeze(self.sp_ps(s), dim=2)
        # m shape -> (batch_size, num_feature)
        return m


class TRNN(nn.Module):
    def __init__(self, sample_length=9, num_hidden=30, num_projection=5):
        super(TRNN, self).__init__()
        self.sample_length = sample_length
        self.num_hidden = num_hidden
        self.num_projection = num_projection

        self.hidden = nn.ModuleList()
        self.tp_rs = nn.Linear(self.num_hidden, self.num_hidden)
        self.tp_vs = nn.Linear(self.num_hidden, self.num_hidden)
        self.tp_bs = nn.Parameter(torch.Tensor(self.num_hidden), requires_grad=True)
        self.activation = nn.ReLU()
        self.tp_projection = nn.Linear(self.sample_length, self.num_projection)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.tp_rs.weight)
        nn.init.zeros_(self.tp_rs.bias)
        nn.init.xavier_normal_(self.tp_vs.weight)
        nn.init.zeros_(self.tp_vs.bias)
        nn.init.zeros_(self.tp_bs)
        nn.init.xavier_normal_(self.tp_projection.weight)
        nn.init.zeros_(self.tp_projection.bias)

    def forward(self, ms):
        # shape of ms -> (t, batch_size, num_features)
        # zero initial hidden
        hiddens = torch.zeros((1, ms.shape[1], self.num_hidden), device=ms.device)
        for t, m_t in enumerate(ms, 1):
            hidden_t = self.activation(self.tp_rs(m_t) + self.tp_vs(hiddens[t-1]) + self.tp_bs)
            hiddens = torch.concatenate((hiddens, hidden_t.unsqueeze(dim=0)), dim=0)
        # shape of hiddens -> (t, batch_size, num_feature)
        # reshape hiddens to (batch_size, num_feature, t)
        q = self.tp_projection(hiddens[1:].permute(1, 2, 0)).permute(0, 2, 1)
        # shape of q -> (batch_size, num_projection, num_feature)
        return q


Sixtytwo_channel_directions = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 58, 52, 53, 54, 55, 56, 62, 60, 61, 51,
     57, 59],
    [59, 57, 51, 61, 60, 62, 56, 55, 54, 53, 52, 58, 50, 41, 32, 23, 14, 49, 40, 31, 22, 13, 48, 39, 30, 21, 12, 47, 38,
     29, 20, 11, 46, 37, 28, 19, 10, 45, 36, 27, 18, 9, 44, 35, 26, 17, 8, 43, 34, 25, 16, 7, 42, 33, 24, 15, 6, 5, 4, 3
        , 2, 1],
    [59, 57, 51, 61, 60, 62, 56, 55, 54, 53, 52, 58, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34,
     33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3
        , 2, 1],
    [1, 2, 3, 4, 5, 6, 15, 24, 33, 42, 7, 16, 25, 34, 43, 8, 17, 26, 35, 44, 9, 18, 27, 36, 45, 10, 19, 28, 37, 46, 11,
     20, 29, 38, 47, 12, 21, 30, 39, 48, 13, 22, 31, 40, 49, 14, 23, 32, 41, 50, 58, 52, 53, 54, 55, 56, 62, 60, 61, 51,
     57, 59]]


# Gets the adjacent hidden layer based on the direction
def n_set(pi, ri, pos, direction):
    point = direction[pi]
    i = pos[point-1][0]
    j = pos[point-1][1]
    neighbors = {
        # 'top-left': [[i, j-1], [i-1, j-1], [i-1, j]],
        # 'top-right': [[i, j+1], [i-1, j], [i-1, j+1]],
        # 'bottom-left': [[i, j-1], [i+1, j-1], [i+1, j]],
        # 'bottom-right': [[i, j-1], [i-1, j-1], [i-1, j]]
        0: [[i, j - 1], [i - 1, j - 1], [i - 1, j]],
        1: [[i, j + 1], [i - 1, j], [i - 1, j + 1]],
        2: [[i, j - 1], [i + 1, j - 1], [i + 1, j]],
        3: [[i, j - 1], [i - 1, j - 1], [i - 1, j]]
    }
    neighbor_set = neighbors[ri]
    neighbor_indexes = []
    for neighbor in neighbor_set:
        if neighbor in pos:
            # location of electrodes
            neighbor_indexes.append(pos.index(neighbor)+1)
    available_neighbor_indexes = []
    for n_idx in neighbor_indexes:
        if n_idx in direction[0:pi]:
            # Returns its index in the specified direction
            available_neighbor_indexes.append(direction.index(n_idx))
    return available_neighbor_indexes


# Two-dimensional coordinates for channels 1 through 62
# https://bcmi.sjtu.edu.cn/home/seed/img/seed-FRA/montage.png
Sixtytwo_channel_coor = [[1, 4], [1, 5], [1, 6], [2, 4], [2, 6], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
                     [3, 8], [3, 9], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [5, 1],
                     [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [6, 1], [6, 2], [6, 3], [6, 4],
                     [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7],
                     [7, 8], [7, 9], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 3], [9, 4], [9, 5],
                     [9, 6], [9, 7]]
