import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torch.optim as optim
from tqdm import tqdm
import yaml

from utils.store import save_state
from utils.metric import Metric

import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, LayerNorm, Dropout

import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from data_utils.constants.deap import HSLT_DEAP_Regions, DEAP_CHANNEL_NAME
from data_utils.constants.seed import HSLT_SEED_Regions, SEED_CHANNEL_NAME

param_path = 'config/model_param/HSLT.yaml'


class HSLT(nn.Module):
    # sample_length, channels, feature_dim, num_classes, get_param, others
    def __init__(self, num_electrodes=32, in_channels=5, num_classes=2, De=8, Dr=16, Le=2, Lr=2):
        super(HSLT, self).__init__()
        self.classes = num_classes
        self.brain_regions = []
        self.De = De
        self.Dr = Dr
        self.Le = Le
        self.Lr = Lr
        self.num_electrodes = num_electrodes
        self.get_param()
        # 定义 Brain Region 的数量
        if num_electrodes == 32:
            self.brain_regions = [
                {"name": "Pre-Frontal", "N": 4},
                {"name": "Frontal", "N": 5},
                {"name": "Left Temporal", "N": 3},
                {"name": "Central", "N": 5},
                {"name": "Right Temporal", "N": 3},
                {"name": "Left Parietal", "N": 3},
                {"name": "Parietal", "N": 3},
                {"name": "Right Parietal", "N": 3},
                {"name": "Occipital", "N": 3},
            ]
            self.regions_electrodes_num = 3
        elif num_electrodes == 62:
            self.brain_regions = [
                {"name": "Pre-Frontal", "N": 5},
                {"name": "Frontal", "N": 9},
                {"name": "Left Temporal", "N": 6},
                {"name": "Central", "N": 7},
                {"name": "Right Temporal", "N": 6},
                {"name": "Left Parietal", "N": 8},
                {"name": "Parietal", "N": 4},
                {"name": "Right Parietal", "N": 7},
                {"name": "Occipital", "N": 10},
            ]
            self.regions_electrodes_num = 4
        self.regions_num = 9
        # Electrode-Level Spatial Learning
        self.transformers = nn.ModuleList()
        for region in self.brain_regions:
            N = region["N"]

            # 定义 LinearEmbedding 和 TransformerEncoder
            patch_embeddings = LinearEmbedding(N, in_channels, self.De)
            patch_transformer = TransformerEncoder(self.De, self.Le)

            if N != self.regions_electrodes_num:
                projection = nn.Linear(N + 1, self.regions_electrodes_num + 1)
            else:
                projection = None

            # 添加到 ModuleList 中
            self.transformers.extend([patch_embeddings, patch_transformer, projection])

        # Brain-Region-Level Spatial Learning
        self.regions_embeddings = LinearEmbedding(self.regions_num, (self.regions_electrodes_num + 1) * self.De,
                                                  self.Dr, False)
        self.regions_transformer = TransformerEncoder(self.Dr, self.Lr)

        # Final prediction layer
        self.prediction = nn.Linear(self.Dr, self.classes)

        if self.classes == 2:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
    def transfer(self, data):
        new_indices = []
        regions = {}
        chan_name = []
        if self.num_electrodes == 32:
            regions = HSLT_DEAP_Regions
            chan_name = DEAP_CHANNEL_NAME
        elif self.num_electrodes == 62:
            regions = HSLT_SEED_Regions
            chan_name = SEED_CHANNEL_NAME
        for key, value in regions.items():
            new_indices.extend([chan_name.index(ele) for ele in value])
        data = data[:,new_indices]
        return data
    def forward(self, inputs):

        transformer_outputs = []
        curr = 0
        for i in range(0, self.regions_num):
            patch_embeddings, patch_transformer, projection = self.transformers[i * 3:(i + 1) * 3]
            N = self.brain_regions[i]["N"]

            # inputs[batch,num_electrodes,d]
            # patch_embeddings:[batch,N,d] -> [batch,1,N+1,De]
            # transformer_output = patch_transformer(patch_embeddings(inputs[:,curr:curr+N,:]))
            x = inputs[:, curr:curr + N, :]
            transformer_output = patch_embeddings(x)
            transformer_output = patch_transformer(transformer_output)
            transformer_output = transformer_output.unsqueeze(1)
            curr = N + curr

            # [batch,1,N+1,De] ->[batch,1,De,N+1] ->[batch,1,De,4]->[batch,1,4,De]
            if N != self.regions_electrodes_num:
                transformer_output = transformer_output.permute(0, 1, 3, 2)
                transformer_output = projection(transformer_output)
                transformer_output = transformer_output.permute(0, 1, 3, 2)

            transformer_outputs.append(transformer_output)

        # Latent features obtained by the 9 transformers
        # 9,[batch,1,4,De]->[batch,9,4,De]
        x = torch.cat(transformer_outputs, dim=1)

        # Brain-Region-Level Spatial Learning
        # [batch,9,4,De]->[batch,9,4*De]
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        # brain_regions_embeddings:[batch,9,4*De]->[batch,10,Dr]
        x = self.regions_embeddings(x)
        regions_outputs = self.regions_transformer(x)

        # Final prediction
        # [batch,1,Dr]
        class_token_output = self.prediction(regions_outputs[:, 0, :])
        class_token_output = self.activation(class_token_output)

        return class_token_output

    def get_param(self):
        try:
            fd = open(param_path, 'r', encoding='utf-8')
            model_param = yaml.load(fd, Loader=yaml.FullLoader)
            fd.close()
            self.De = model_param['params']['De']
            self.Dr = model_param['params']['Dr']
            self.Le = model_param['params']['Le']
            self.Lr = model_param['params']['Lr']
            print("\nUsing setting from {}\n".format(param_path))
        except IOError:
            print("\n{} may not exist or not available".format(param_path))

        print("HSLT Model, Parameters:\n")
        print("{:45}{:20}".format("De is input embedding dimension (electrode level)", self.De))
        print("{:45}{:20}".format("Dr is embedding dimension (brain - region level):", self.Dr))
        print("{:45}{:20}".format("Le is no of encoders (electrode level)", str(self.Le)))
        print("{:45}{:20}\n".format("Lr is no of encoder (brain - region level)", self.Lr))
        if self.De != 8 or self.Dr != 16 or self.Le != 2 or self.Lr != 2:
            print("Not Using Default Setting, the performance may be not the best")
        print("Starting......")

# Electrode Patch encoder
class LinearEmbedding(nn.Module):
    def __init__(self, num_patches, in_channels, projection_dim, expand=True, regions_dropout=0.1):
        super(LinearEmbedding, self).__init__()
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.projection_dim = projection_dim
        self.expand = expand

        # Create class token,将 class_token 声明为模型参数
        self.class_token = nn.Parameter(torch.randn(1, projection_dim), requires_grad=True)

        # Dense layer for linear transformation of electrode patches (Map to constant size De)
        self.projection = nn.Linear(in_channels, projection_dim)

        # Embedding layer for positional embeddings
        self.position_embedding = nn.Embedding(num_patches + 1, projection_dim)

        self.dropout = nn.Dropout(regions_dropout)

    def forward(self, patch):
        if self.expand:  # For electrode-level spatial learning
            # expand dimension 1, so that we can stack the transformer outputs in the brain-region-level

            # patch = patch.unsqueeze(1)

            # get batch_size
            batch = patch.size(0)
            # augment class token's first dimension to match the batch_size
            class_token = self.class_token.expand(batch, -1)
            # reshape the class token to match patches dimensions
            # from (batch, De) to (batch, 1, 1, De)
            class_token = class_token.view(batch, 1, self.projection_dim)
            # calculate patch embeddings
            patches_embed = self.projection(patch)
            # shape: (None, 1, N, De)
            patches_embed = torch.cat([class_token, patches_embed], 1)
            # calculate position embeddings
            positions = torch.arange(0, self.num_patches + 1, 1, device=patch.device)

            positions_embed = self.position_embedding(positions)
            # Add positions to patches
            encoded = patches_embed + positions_embed

        else:  # For brain-region-level spatial learning
            # we do the same as before, but we don't expand dimensions; it's already stacked (concat) on top of each other
            batch = patch.size(0)
            class_token = self.class_token.expand(batch, -1)
            class_token = class_token.view(batch, 1, self.projection_dim)
            patches_embed = self.projection(patch)
            patches_embed = self.dropout(patches_embed)
            patches_embed = torch.cat([class_token, patches_embed], 1)
            # calculate position embeddings
            positions = torch.arange(0, self.num_patches + 1, 1, device=patch.device)
            positions_embed = self.position_embedding(positions)
            # Add positions to patches
            encoded = patches_embed + positions_embed

        return encoded


# Hyperparameters (remain constant through the whole model)
# dropout_rate = 0.4  # Dropout rate
# Dh = 64  # dimension of weights (MSA)
# k = 16  # num of heads in MSA

# !!!!!!!! The original ViT paper (and Attention is all you need) suggest Dh to always be equal to De/k !!!!!!!!!!!!!!!
# And here they don't apply that rule !!!!!!

# MLP layer
class MLP(nn.Module):
    def __init__(self, hidden_states, output_states, dropout=0.4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(output_states, hidden_states)
        self.fc2 = nn.Linear(hidden_states, output_states)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.gelu(self.fc1(x))
        dr_hidden = self.dropout(hidden)
        output = F.gelu(self.fc2(dr_hidden))
        dr_output = self.dropout(output)
        return dr_output


# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads=16, msa_dimensions=64, dropout_rate=0.4):
        super(TransformerEncoderBlock, self).__init__()
        self.model_dim = model_dim
        self.layernormalization1 = nn.LayerNorm(model_dim, eps=1e-6)
        self.fc1 = nn.Linear(model_dim, msa_dimensions)
        self.attention = nn.MultiheadAttention(embed_dim=msa_dimensions, num_heads=num_heads, dropout=dropout_rate,
                                               batch_first=True)
        self.fc2 = nn.Linear(msa_dimensions, model_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=2,kdim=4, dropout=dropout_rate)
        self.layernormalization2 = nn.LayerNorm(model_dim, eps=1e-6)
        self.mlp = MLP(model_dim * 4, model_dim)

    def forward(self, x):
        # layer normalization 1.
        x1 = self.layernormalization1(x)
        # create a multi-head attention layer.

        x1 = F.relu(self.fc1(x1))
        attention_output, _ = self.attention(x1, x1, x1)
        attention_output = F.relu(self.fc2(attention_output))

        x2 = x + attention_output
        x3 = self.layernormalization2(x2)
        x3 = self.mlp(x3)
        y = x2 + x3
        return y


#  Transformer Encoder Block x L Repeat
class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_blocks):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(TransformerEncoderBlock(model_dim))

    def forward(self, x):
        # create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        return x

# model = HierarchicalTransformer(classes, N, d, De, Dr, Le, Lr)
# input_data = torch.randn((batch_size, N, d))  # Replace batch_size with your actual batch size
# output = model(input_data)
#
# # 为了适应数据的尺寸和维度，您可能需要调整模型的一些细节
#
# # 数据加载
# with open("deap_hvlv_x", "rb") as fp:
#     x = torch.tensor(pickle.load(fp))
#
# with open("deap_hvlv_y", "rb") as fp:
#     y = torch.tensor(pickle.load(fp))
#
# # ...（其他辅助函数）
#
# # 检查是进行二分类还是四分类
# classes = 2
# if y[0].max() > 1:
#     classes = 4

# # 模型
# class HierarchicalTransformer(nn.Module):
#     def __init__(self, d, De, Dr, Le, Lr, classes):
#         super(HierarchicalTransformer, self).__init__()
#
#         # 在这里定义您的模型架构
#
#     def forward(self, input_data):
#         # 定义模型的前向传播
#
# # 创建模型
# model = HierarchicalTransformer(d, De, Dr, Le, Lr, classes)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.005)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps)
#
# # 训练模型
# epochs = 80
# batch_size = 512
# loo = LeaveOneOut()
# average_results_acc = []
# average_results_f1 = []
# average_results_cohen = []
#
# for train_index, test_index in loo.split(x):
#     print('-----------------------------------------')
#     print('count = ' + str(test_index[0]))
#
#     # 数据准备
#     train = TensorDataset(x[train_index], y[train_index])
#     test = TensorDataset(x[test_index], y[test_index])
#     train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
#
#     # 训练模型
#     for epoch in range(epochs):
#         model.train()
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#         scheduler.step()
#
#     # 评估模型
#     model.eval()
#     with torch.no_grad():
#         # 在这里执行评估

