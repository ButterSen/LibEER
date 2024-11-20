import torch
import torch.nn as nn
import torch.utils.data

import numpy as np
import pandas as pd

param_path = 'config/model_param/BiDANN.yaml'

SEED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4','F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
LEFT_CHANNEL_NAME = ['FP1', 'AF3', 'F7', 'F5', 'F3',
'F1','FT7', 'FC5', 'FC3', 'FC1', 'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3',
'CP1', 'P7', 'P5', 'P3', 'P1', 'PO7', 'PO5', 'PO3', 'CB1', 'O1', 'FPZ', 'FCZ',
'CPZ', 'POZ']
RIGHT_CHANNEL_NAME = ['FP2', 'AF4', 'F8',
'F6', 'F4', 'F2', 'FT8', 'FC6', 'FC4', 'FC2', 'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6',
'CP4', 'CP2', 'P8', 'P6', 'P4', 'P2', 'PO8', 'PO6', 'PO4', 'CB2', 'O2', 'FZ', 'CZ',
'PZ', 'OZ']

# A Bi-Hemisphere Domain Adversarial Neural Network Model for EEG Emotion Recognition
# BiDANN paper link : https://ieeexplore.ieee.org/document/8567966
# Y. Li, W. Zheng, Y. Zong, Z. Cui, T. Zhang and X. Zhou, "A Bi-Hemisphere Domain Adversarial Neural Network Model for EEG Emotion Recognition," in IEEE Transactions on Affective Computing, vol. 12, no. 2, pp. 494-504, 1 April-June 2021, doi: 10.1109/TAFFC.2018.2885474.


class BiDANN(nn.Module):
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3, sample_length=9, 
                 domain_classes = 2, lambda_= 1, device = None):
        # num_electrodes(int): The number of electrodes.s=1,
        # in_channels(int): The feature dimension of each electrode.
        # num_classes(int): The number of emotions to predict.
        super(BiDANN, self).__init__()
        
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        self.domain_classes = domain_classes
        self.lambda_ = lambda_
        self.length = sample_length
        self.device = device
       
        #2 feature extractor， 1 classifier， 2 local discriminators， 1 global discriminator
        self.l_extractor = BiDANNFeatureExtractor(num_electrodes=self.num_electrodes, in_channels=self.in_channels, sample_length=self.length, hidden_size = 128,output_size=8)
        self.r_extractor = BiDANNFeatureExtractor(num_electrodes=self.num_electrodes, in_channels=self.in_channels, sample_length=self.length, hidden_size = 128,output_size=8)
        self.classifer = BiDANNClassifer(input_size=8*128, hidden_size1=256, hideen_size2=64,num_classes=3)
        self.r_discriminator = LocalDiscriminator(input_size = 8*128, hidden_size1 = 256, hidden_size2 = 64,domain_classes = self.domain_classes, lambda_=self.lambda_)
        self.l_discriminator = LocalDiscriminator(input_size = 8*128, hidden_size1 = 256, hidden_size2 = 64,domain_classes = self.domain_classes, lambda_=self.lambda_)
        self.global_discriminator = GlobalDiscriminator(input_size = 8*128, hidden_size1 = 256, hidden_size2 = 64,domain_classes = self.domain_classes, lambda_=self.lambda_)
    
    def forward(self, source_data, target_data):
        # x: (batch_size, sample_length, num_electrodes, in_channels)
        # get left and right features  
        l_source_data, r_source_data = divide_r_l(source_data,columns=SEED_CHANNEL_NAME,l_columns=LEFT_CHANNEL_NAME, r_columns=RIGHT_CHANNEL_NAME)
        l_target_data, r_target_data = divide_r_l(target_data,columns=SEED_CHANNEL_NAME,l_columns=LEFT_CHANNEL_NAME, r_columns=RIGHT_CHANNEL_NAME)
        l_source_data = l_source_data.to(self.device)
        r_source_data = r_source_data.to(self.device)
        l_target_data = l_target_data.to(self.device)
        r_target_data = r_target_data.to(self.device)  

        # extract features
        l_source_feature = self.l_extractor(l_source_data)
        r_source_feature = self.r_extractor(r_source_data)
        l_target_feature = self.l_extractor(l_target_data)
        r_target_feature = self.r_extractor(r_target_data)

        #Classify
        label_predict = self.classifer(l_source_feature, r_source_feature)

        #Domain Discriminate
        l_domain_predict = self.l_discriminator(l_source_feature, l_target_feature)
        r_domain_predict = self.r_discriminator(r_source_feature, r_target_feature)
        global_domain_predict = self.global_discriminator(l_source_feature, r_source_feature, l_target_feature, r_target_feature)

        #print(label_predict[0], l_domain_predict[0], r_domain_predict[0], global_domain_predict[0])
        #result
        return label_predict, l_domain_predict, r_domain_predict, global_domain_predict

#Dividing the features into left and right parts
def divide_r_l(features,columns,l_columns, r_columns):#batch_size*9*62*5
    features = features.transpose(2, 3)
    features_reshaped = features.reshape(features.shape[0]*features.shape[1]*features.shape[2], -1)
    features_reshaped_copy = features_reshaped.cpu()
    features_array = features_reshaped_copy.numpy()
    df_features = pd.DataFrame(features_array, columns = columns)
    
    df_left = df_features[l_columns]
    df_right = df_features[r_columns]
    
    left_features = torch.tensor(np.array(df_left))
    left_features = left_features.reshape(features.shape[0], features.shape[1],features.shape[2], -1)
    left_features = left_features.transpose(2,3)
    left_features = left_features.reshape(features.shape[0], features.shape[1], -1)

    right_features = torch.tensor(np.array(df_right))
    right_features = right_features.reshape(features.shape[0], features.shape[1], features.shape[2], -1)
    right_features = right_features.transpose(1,2)
    right_features = right_features.reshape(features.shape[0], features.shape[1], -1)
    return left_features, right_features #batchh_size*9*155


class BiDANNFeatureExtractor(nn.Module):#Input: batch_size*9*155, 
    def __init__(self, num_electrodes=62, in_channels=5, sample_length=9, hidden_size = 128,output_size=8):
        super(BiDANNFeatureExtractor, self).__init__()
        self.input = num_electrodes * in_channels // 2
        self.out_lstm = hidden_size 
        self.length = sample_length
        self.output = output_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=self.input, hidden_size=self.out_lstm, num_layers=2, batch_first=True, dropout = 0.3)
        self.linear = nn.Linear(in_features = self.length, out_features = self.output)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.initialize()

    def initialize(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Extract features
        x, _ = self.lstm(x)  # output of LSTM:(x, (h_n, c_n)), we only need x, batch_size*9*128
        x = x.transpose(1, 2) 
        x = self.linear(x)         
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.transpose(1,2)
        x= x.reshape(-1, self.output*self.out_lstm)
        return x #Output:batch_size*(8*128)
 
class BiDANNClassifer(nn.Module):#Input: batch_size*8*128
    def __init__(self,input_size=3*128, hidden_size1=128, hideen_size2=32,num_classes=3):
        super(BiDANNClassifer,self).__init__()
        self.input = input_size
        self.hidden1 = hidden_size1
        self.hidden2 = hideen_size2
        self.num_classes = num_classes

        self.classifer = nn.Sequential(
            nn.Linear(in_features = 2*self.input, out_features = 2*self.hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(2*self.hidden1),
            nn.Linear(in_features = 2*self.hidden1, out_features = 2*self.hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(2*self.hidden2),
            nn.Linear(in_features = 2*self.hidden2, out_features = self.num_classes),
        )       
        self.initialize()

    def initialize(self):
        for m in self.classifer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  #He Initialization，suitable for relu activation function
                nn.init.zeros_(m.bias)  
        
    def forward(self, l_feature, r_feature):
        source_feature =  torch.cat((l_feature, r_feature), dim = 1)
        source_feature = self.classifer(source_feature)
        #Output: batch_size*3， corresponding to the predicted emotion label for each sample in the batch.
        return source_feature
    

class ReverseGrad(torch.autograd.Function):#Reverse the gradient of the input tensor
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambda_ * grad_out, None


class LocalDiscriminator(nn.Module):#Input: batch_size*8*128
    def __init__(self,input_size = 3*128, hidden_size1 = 128, hidden_size2 = 32,domain_classes = 2, lambda_=0.5):
        super(LocalDiscriminator,self).__init__()
        self.lambda_ = lambda_
        self.input_size = input_size
        self.hidden1 = hidden_size1
        self.hidden2 = hidden_size2
        self.domain_classes = domain_classes
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_size, self.hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden1),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden2),
            nn.Linear(self.hidden2, self.domain_classes),
        )
        self.initialize()

    def initialize(self):
        for m in self.discriminator:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  #He Initialization，suitable for relu activation function
                nn.init.zeros_(m.bias)  

    def forward(self, feature1, feature2):
        features = torch.cat((feature1, feature2), dim = 0)#(2*batch_size)*(8*128)
        reversed_features = ReverseGrad.apply(features, self.lambda_)
        output = self.discriminator(reversed_features)
        return output#Output:（2*batch_size)*2, corresponding to the predicted domain label for each sample in the batch.

class GlobalDiscriminator(nn.Module):
    def __init__(self,input_size = 3*128 ,hidden_size1 = 128, hidden_size2 = 32, domain_classes = 2, lambda_=0.5):
        super(GlobalDiscriminator,self).__init__()
        self.lambda_ = lambda_
        self.input_size = input_size
        self.hidden1= hidden_size1
        self.hidden2 = hidden_size2
        self.output = domain_classes
        self.discriminator = nn.Sequential(
            nn.Linear(2*self.input_size, 2*self.hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(2*self.hidden1),
            nn.Linear(2*self.hidden1, 2*self.hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(2*self.hidden2),
            nn.Linear(2*self.hidden2, self.output),
        )
        self.initialize()

    def initialize(self):
        for m in self.discriminator:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  
                nn.init.zeros_(m.bias)  

    def forward(self, l_source, r_source, l_target, r_target):
        source_features = torch.cat((l_source, r_source), dim = 1)
        target_features = torch.cat((l_target, r_target), dim = 1)
        all_features = torch.cat((source_features, target_features), dim = 0)#(2*batch_size)*(2*8*128) 
        reversed_features = ReverseGrad.apply(all_features, self.lambda_)
        output = self.discriminator(reversed_features)#(2*batch_size)*2      
        return output


    

    
