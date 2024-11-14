# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np 
from copy import deepcopy
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm 

from utils.store import save_state
from utils.metric import Metric
from data_utils.preprocess import normalize

def square(x):
    return x * x

def cov(x): 
    x_t = x.permute([0, 1, 3, 2])
    return torch.matmul(x_t, x)

def safe_log(x, eps=1e-6):
    return torch.log(torch.clamp(x, min=eps))
  
class Expression(nn.Module):

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )

class Conv2dNormWeight(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dNormWeight, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dNormWeight, self).forward(x)


class CNN(nn.Module):
    def __init__(self,ic,ih,iw,kh,kw,ks,ph,pw,ps,oc):
        super(CNN,self).__init__()
        # input
        self.input_channel = ic
        self.input_height =  ih
        self.input_width = iw
        self.output_channel = oc
        self.kernel_height = kh
        self.kernel_width = kw
        self.kernel_stride = ks
        self.pooling_height = ph
        self.pooling_width = pw
        self.pooling_stride = ps
        # CNN
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channel,self.output_channel,(self.kernel_height,self.kernel_width),self.kernel_stride),
            nn.ELU(),
            nn.MaxPool2d((self.pooling_height,self.pooling_width),self.pooling_stride)
        )
        # dropout
        self.dropout = nn.Dropout2d(p=0.5)
    def __call__(self,x):
        x = x.permute(0,1,3,2)
        c = self.conv(x)
        # c1 = c.reshape(800,-1)
        cd = self.dropout(c)
        return cd
    
class channel_wise_attention(nn.Module):
    def __init__(self,H,W,C,reduce):
        super(channel_wise_attention,self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.r = reduce
        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(self.C,self.r),
            nn.Tanh(),
            nn.Linear(self.r,self.C)
        )
        # softmax
        self.softmax = nn.Softmax(dim=3)

    def forward(self,x):
        # mean pooling
        x1 = x.permute(0,3,1,2)
        mean = nn.AvgPool2d((1,self.W))
        feature_map = mean(x1).permute(0,2,3,1)
        # FC Layer
        # feature_map : [800,1,1,C]
        feature_map_fc = self.fc(feature_map)
        
        # softmax
        v = self.softmax(feature_map_fc)
        # channel_wise_attention
        v = v.reshape(-1,self.C)
        vr = torch.reshape(torch.cat([v]*(self.H*self.W),axis=1),[-1,self.H,self.W,self.C])
        channel_wise_attention_fm = x * vr
        return v, channel_wise_attention_fm
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        # lstm
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
            )

    def forward(self,x,hidden0=None):
        x = x.reshape(-1,1,self.input_size)
        q ,(hidden,cell) = self.lstm(x)
        h = hidden[1].reshape(-1,1,64)
        c = cell[1].reshape(-1,1,64)
        return h,c
    
class dense(nn.Module):
    def __init__(self,input_dim1,input_dim2,hidden_dim,
    activation=lambda x:x):
        super().__init__()
        self.W1 = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim1,hidden_dim))))
        self.W2 = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim2,hidden_dim))))
        self.b = nn.Parameter(torch.Tensor(np.zeros(hidden_dim)))
        self.activation = activation
        self.vector = nn.Linear(input_dim2,input_dim2)
    def forward(self,x):
        y = self.vector(x)
        return self.activation(torch.matmul(x,self.W1)+torch.matmul(y,self.W2)+self.b)

class self_attention(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(self_attention,self).__init__()
        self.q = input_dim
        self.k = input_dim
        self.hidden = hidden_dim
        self.dense = dense(self.q,self.k,self.k)
        self.self_attention = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.k,self.k)
        )
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout()
    
    def forward(self,x):
        # print(x.shape)
        y = self.dense(x)
        # print(y.shape)
        z = self.self_attention(y)
        #print(z.shape)
        p = z * x
        p = self.softmax(p)
        #print(p.shape)
        A = p * x
        #print(A.shape)
        A = A.reshape(-1,self.k)
        A = self.dropout(A)
        return A        
    
class ACRNN(nn.Module): #  model = Model[args.model](args.sample_length, channels, feature_dim, num_classes)
    def __init__(self, n_channels, n_timepoints, num_classes):
        super(ACRNN,self).__init__()
        self.H = 1
        self.W = n_timepoints
        self.C = n_channels
        self.reduce = 15
        self.channel_wise_attention = channel_wise_attention(self.H,self.W,self.C,self.reduce)
        self.output_channel = 40
        self.kernel_height = n_channels
        self.kernel_width = 45
        self.kernel_stride = 1
        self.pooling_height = 1
        self.pooling_width = 75
        self.pooling_stride = 10
        self.cnn = CNN(self.H,self.C,self.W,self.kernel_height,self.kernel_width,self.kernel_stride,self.pooling_height,self.pooling_width,self.pooling_stride,self.output_channel)
        self.hidden_dim = 64
        c_width = int((((n_timepoints - self.kernel_width)/self.kernel_stride+1)-self.pooling_width)/self.pooling_stride + 1)
        # print("c_width: {}".format(c_width))
        self.lstm = LSTM(self.output_channel * c_width, self.hidden_dim)
        self.hidden = 512
        self.self_attention = self_attention(self.hidden_dim,self.hidden)
        self.num_labels = num_classes
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_dim,self.num_labels),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x_map, x_ca = self.channel_wise_attention(x)
        x_cn = self.cnn(x_ca)
        x_rn, x_c = self.lstm(x_cn)
        x_sa = self.dropout(self.self_attention(x_rn))
        x_sm = self.softmax(x_sa)
        return x_sm


