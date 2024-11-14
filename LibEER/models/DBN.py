
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
from tqdm import tqdm
from data_utils.preprocess import normalize
from utils.store import save_state
from utils.metric import Metric
param_path = 'config/model_param/DBN.yaml'
import yaml
#RBM类的定义
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units))
        self.v_bias = nn.Parameter(torch.randn(visible_units))
        self.h_bias = nn.Parameter(torch.randn(hidden_units))
        self.w_momentum = torch.zeros(visible_units, hidden_units)

    def sample_h(self, v):
        prob_hidden=torch.matmul(v,self.W)
        prob_hidden=prob_hidden+self.h_bias
        prob_hidden=torch.sigmoid(prob_hidden)
        return prob_hidden, torch.bernoulli(prob_hidden)

    def sample_v(self, h):
        prob_visible = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        prob_visible_gauss = prob_visible + torch.randn_like(prob_visible)
        return prob_visible, prob_visible_gauss

    def forward(self, v):
        p_h_given_v = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        h_sample = torch.bernoulli(p_h_given_v)
        p_v_given_h = torch.sigmoid(torch.matmul(h_sample, self.W.t()) + self.v_bias)
        return p_v_given_h, p_h_given_v

    def constrastive_divergence(self, v,batch_size,device, learning_rate=0.005, momentum=0.1):
        v = v.to(device)
        self.W = self.W.to(device)
        self.v_bias = self.v_bias.to(device)
        self.h_bias = self.h_bias.to(device)
        self.w_momentum = self.w_momentum.to(device)
        positive_hidden_prob, positive_hidden = self.sample_h(v)
        positive_association = torch.matmul(v.t(), positive_hidden_prob)
        hidden = positive_hidden
        visible_prob, visible = self.sample_v(hidden)
        hidden_prob, hidden = self.sample_h(visible_prob)
        negative_visible_prob = visible
        negative_hidden_prob = hidden_prob
        negative_association = torch.matmul(negative_visible_prob.t(), negative_hidden_prob)

        self.w_momentum *= momentum
        self.w_momentum += (positive_association - negative_association)

        self.W.data.add_(self.w_momentum * learning_rate / batch_size)
        self.v_bias.data.add_(learning_rate * torch.sum(v - visible, dim=0) / batch_size)
        self.h_bias.data.add_(learning_rate * torch.sum(positive_hidden_prob - negative_hidden_prob, dim=0) / batch_size)

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))


#DBN
class DBN(nn.Module):
    def __init__(self,num_electrodes=62, in_channels=5, num_classes=3,hidden_size1=300,hidden_size2=400):
        super(DBN,self).__init__()
        self.num_electrodes=num_electrodes
        self.in_channels=in_channels
        self.num_classes=num_classes
        #not used
        #self.get_param()
        self.rbm1=RBM(self.num_electrodes * self.in_channels,hidden_size1)
        self.rbm2=RBM(hidden_size1,hidden_size2)
        self.fc=nn.Linear(hidden_size2,num_classes)

    def get_param(self):
        try:
            fd = open(param_path, 'r', encoding='utf-8')
            model_param = yaml.load(fd, Loader=yaml.FullLoader)
            fd.close()
            self.hidden_size1 = model_param['params']['h1']
            self.hidden_size2 = model_param['params']['h2']
            print("\nUsing setting from {}\n".format(param_path))
        except IOError:
            print("\n{} may not exist or not available".format(param_path))

        print("DBN Model, Parameters:\n")
        print("{:45}{:20}".format("h1:", self.hidden_size1))
        print("{:45}{:20}".format("h2:", self.hidden_size2))
        print("Starting......")

    def forward(self,v):
        #v=v.view(v.shape[0],-1).type(torch.FloatTensor)
        h1_prob,h1=self.rbm1(v)
        h2_prob,h2=self.rbm2(h1)
        output=self.fc(h2)
        return output
    #reconstruction
    def reconstruct(self,v,device):
        h0=v
        # forward pass through rbms to get hidden representation前向传播
        for rbm_layer in[self.rbm1,self.rbm2]:
            h0.to(device)
            p_h,h0=rbm_layer.sample_h(h0)
        v0=h0
        # backward pass through rbms to reconstruct visible representation后相传播
        for rbm_layer in [self.rbm2,self.rbm1]:
            v0.to(device)
            p_v,v0=rbm_layer.sample_v(h=v0)
        return v0
