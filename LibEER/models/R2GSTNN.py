import torch
import torch.nn as nn
import torch.utils.data

REGION_INDEX = [[3,0,1,2,4],[7,8,9,10,11],[5,6],[13,12],[14,15,23,24,32,33],
                [22,21,31,30,40,39],[16,17,18,19,20],[25,26,27,28,29],
                [34,35,36,37,38],[41,42],[49,48],[43,44,45,46,47],
                [50,51,57],[56,55,61],[52,53,54],[58,59,60]]

class R2GSTNN(nn.Module):
    def __init__(self, input_size=5,  num_classes=3, regions=16, region_index=REGION_INDEX, k=3, t=9,
                 regional_size=100, global_size = 150,regional_temporal_size=200, global_temporal_size=250,
                 domain_classes=2, lambda_ = 1,dropout=0.5):
        super(R2GSTNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.regions = regions
        self.region_index = region_index
        self.k = k
        self.t = t
        self.regional_size = regional_size
        self.global_size = global_size
        self.regional_temporal_size = regional_temporal_size
        self.global_temporal_size = global_temporal_size
        self.domain_classes = domain_classes
        self.lambda_ = lambda_
        self.dropout = dropout

        self.regional_learner = RegionFeatureLearner(input_size=self.input_size, regional_size=self.regional_size, regions=self.regions, region_index=self.region_index)
        self.regional_attention = RegionAttention(regional_size=self.regional_size, regions=self.regions)
        self.global_learner = GlobalFeatureLearner(regional_size=self.regional_size, global_size=self.global_size, regions=self.regions, k=self.k)
        self.temporal_learner = TemporalFeatureLearner(k=self.k, t=self.t, regions=self.regions, regional_size=self.regional_size, global_size=self.global_size, 
                                                      regional_temporal_size=self.regional_temporal_size, global_temporal_size=self.global_temporal_size, dropout=self.dropout)
        self.classifer = Classifer(regions=self.regions, regional_temporal_size=self.regional_temporal_size, global_temporal_size=self.global_temporal_size, 
                                   num_classes=self.num_classes,hidden_size1=512, hidden_size2=128)
        self.discriminator = Discriminator(regions=self.regions, regional_temporal_size=self.regional_temporal_size, global_temporal_size=self.global_temporal_size, 
                                           domain_classes=self.domain_classes, lambda_=self.lambda_, hidden_size1=512, hidden_size2=128)
        
    def forward(self, source_data, target_data):
        #source_data: (batch_size, T, num_electrodes, d)
        #target_data: (batch_size, T, num_electrodes, d)
        source_regional_feature = self.regional_learner(source_data)
        source_attention_feature = self.regional_attention(source_regional_feature)
        source_global_feature = self.global_learner(source_attention_feature)
        source_temporal_feature = self.temporal_learner(source_regional_feature, source_global_feature)
        source_label_prediction = self.classifer(source_temporal_feature)

        target_regional_feature = self.regional_learner(target_data)
        target_attention_feature = self.regional_attention(target_regional_feature)
        target_global_feature = self.global_learner(target_attention_feature)
        target_temporal_feature = self.temporal_learner(target_regional_feature, target_global_feature)
        
        domain_prediction = self.discriminator(source_temporal_feature, target_temporal_feature)

        return source_label_prediction, domain_prediction


class RegionFeatureLearner(nn.Module):#input: (batch_size*T, num_electrodes, d)
    def __init__(self, input_size=5, regional_size=100, regions=16, region_index=REGION_INDEX): 
        super(RegionFeatureLearner, self).__init__()
        self.regions = regions
        self.input_size = input_size
        self.regional_size = regional_size
        self.region_index = [torch.LongTensor(e) for e in region_index]

        self.bilstm = nn.ModuleList([nn.LSTM(self.input_size, self.regional_size, batch_first=True, bidirectional=True) for i in range(regions)])

    def forward(self, features):
        regional_feature_input =[]
        regional_feature_list = []
        features = features.reshape(-1, features.shape[2], features.shape[3])
        for i in range(self.regions):
            regional_feature_input.append(features[:,self.region_index[i],:])
            hidden_unit = (self.bilstm[i](regional_feature_input[i])[0])
            regional_feature_list.append(hidden_unit[:, -1, :].unsqueeze(1))#keep the dimension
        
        regional_feature = torch.cat(regional_feature_list, dim=1)
        #regional_feature: (batch_size*T, regions, 2*hidden_size)
        return regional_feature
    
class RegionAttention(nn.Module):
    def __init__(self, regional_size=100, regions=16):
        super(RegionAttention, self).__init__()
        self.regional_size = regional_size
        self.regions = regions

        self.P = nn.Parameter(torch.Tensor(2*self.regional_size, self.regions))
        self.tanh = nn.Tanh()
        self.bias = nn.Parameter(torch.Tensor(self.regions))
        self.Q = nn.Parameter(torch.Tensor(self.regions, self.regions))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, regional_feature):
        #regional_feature: (batch_size*T, regions, 2*regional_hidden_size)
        W = self.softmax(torch.matmul(self.tanh(torch.matmul(regional_feature, self.P) + self.bias), self.Q))
        #W: (batch_size*T, regions, regions)
        regional_feature = regional_feature.transpose(1,2)
        attention_feature = torch.matmul(regional_feature, W)
        attention_feature = attention_feature.transpose(1,2)
        #attention_feature: (batch_size*T, regions, 2*regional_hidden_size)
        return attention_feature
    
class GlobalFeatureLearner(nn.Module):
    def __init__(self, regional_size=100, global_size = 150, regions=16, k=3 ):
        super(GlobalFeatureLearner, self).__init__()
        self.regional_size = regional_size
        self.global_size = global_size
        self.regions = regions
        self.k = k
        
        self.bilstm = nn.LSTM(input_size=2*self.regional_size, hidden_size=self.global_size//2, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(self.regions, self.k)
        self.relu = nn.ReLU()

    def forward(self, attention_feature):
        #attention_feature: (batch_size*T, regions, 2*regional_hidden_size)
        hidden_unit = self.bilstm(attention_feature)[0]
        hidden_unit = hidden_unit.transpose(1,2)
        global_feature = self.fc2(hidden_unit)
        global_feature = self.relu(global_feature)

        #global_feature: (batch_size*T, global_hidden_size, k)
        return global_feature
    
class TemporalFeatureLearner(nn.Module):
    def __init__(self, k=3, t=9,regions=16,regional_size=100, global_size=150,  
                  regional_temporal_size=200,global_temporal_size=250, dropout = 0.5 ):
        super(TemporalFeatureLearner, self).__init__()
        self.k = k
        self.t = t
        self.regions = regions
        self.regional_size = regional_size
        self.global_size = global_size
        self.regional_temporal_size = regional_temporal_size
        self.global_temporal_size = global_temporal_size

        self.dropout = nn.Dropout(dropout)
        self.regional_bilstm = nn.LSTM(input_size=2*self.regional_size, hidden_size=self.regional_temporal_size//2, batch_first=True, bidirectional=True)
        self.global_bilstm = nn.LSTM(input_size = self.global_size*self.k, hidden_size=self.global_temporal_size//2, batch_first=True, bidirectional=True)

    def forward(self, regional_feature, global_feature):
        #regional_feature: (batch_size*T, regions, 2*regional_hidden_size)
        #global_feature: (batch_size*T, global_hidden_size, k)
        regional_feature = regional_feature.reshape(-1, self.t, self.regions, 2*self.regional_size)
        regional_feature = regional_feature.transpose(1,2)
        regional_feature = regional_feature.reshape(-1, self.t, 2*self.regional_size)
        regional_feature = self.regional_bilstm(regional_feature)[0]
        regional_feature = regional_feature[:, -1, :]
        regional_temporal_feature = regional_feature.reshape(-1, self.regions*self.regional_temporal_size)

        global_feature  = global_feature.reshape(-1, self.t, self.global_size*self.k )
        global_feature =  self.global_bilstm(global_feature)[0]
        global_feature = global_feature[:, -1, :]
        global_temporal_feature = global_feature.reshape(-1, self.global_temporal_size)

        global_regional_temporal_feature = torch.cat([global_temporal_feature, regional_temporal_feature], dim=1)
        global_regional_temporal_feature = self.dropout(global_regional_temporal_feature)
        
        #global_regional_temporal_feature: (batch_size, global_hidden_size+regional_hidden_size*regions)
        return global_regional_temporal_feature


class Classifer(nn.Module):
    def __init__(self, regions = 16, regional_temporal_size = 200, global_temporal_size = 250, num_classes = 3,
                 hidden_size1 =512, hidden_size2 = 128):
        super(Classifer, self).__init__()
        self.regions = regions
        self.regional_temporal_size = regional_temporal_size
        self.global_temporal_size = global_temporal_size
        self.num_classes = num_classes
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.classifer = nn.Sequential(
                nn.Linear(in_features=self.global_temporal_size+self.regions*self.regional_temporal_size, out_features=512),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size1),
                nn.Linear(in_features=self.hidden_size1, out_features=self.hidden_size2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size2),
                nn.Linear(in_features=self.hidden_size2, out_features=self.num_classes)
        )
    
    def forward(self, global_regional_temporal_feature):
        #global_regional_temporal_feature: (batch_size, global_hidden_size+regional_hidden_size*regions)
        label_prediction = self.classifer(global_regional_temporal_feature)
        #label_prediction: (batch_size, num_classes)
        return label_prediction

class ReverseGrad(torch.autograd.Function):#Reverse the gradient of the input tensor
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambda_ * grad_out, None

class Discriminator(nn.Module):
    def __init__(self, regions = 16, regional_temporal_size=200, global_temporal_size=250,domain_classes=2,lambda_=1,
                  hidden_size1=512, hidden_size2=128):
        super(Discriminator, self).__init__()
        self.regions = regions
        self.regional_temporal_size = regional_temporal_size
        self.global_temporal_size = global_temporal_size
        self.domain_classes = domain_classes
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lambda_ = lambda_

        self.discriminator = nn.Sequential(
                nn.Linear(in_features=self.global_temporal_size+self.regions*self.regional_temporal_size, out_features=512),
                nn.ReLU(),
                #nn.BatchNorm1d(self.hidden_size1),
                nn.Linear(in_features=self.hidden_size1, out_features=self.hidden_size2),
                nn.ReLU(),
                #nn.BatchNorm1d(self.hidden_size2),
                nn.Linear(in_features=self.hidden_size2, out_features=self.domain_classes)
        )

    def forward(self, source_feature, teaget_feature):
        #source_feature: (batch_size, global_hidden_size+regional_hidden_size*regions)
        #target_feature: (batch_size, global_hidden_size+regional_hidden_size*regions)
        features = torch.cat([source_feature, teaget_feature], dim=0)
        reversed_features = ReverseGrad.apply(features, self.lambda_)
        domain_prediction = self.discriminator(reversed_features)
        #domain_prediction: (2*batch_size, domain_classes)
        return domain_prediction
    

