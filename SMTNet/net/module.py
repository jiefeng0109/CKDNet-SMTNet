import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from .utils import convolution
# from torch_geometric.nn import global_mean_pool,DNAConv,GATConv
# from torch_geometric.data import Data,DataLoader
from config import system_configs
# class Graph_Net(torch.nn.Module):
#     def __init__(self,
#                  in_channels,
#                  hidden_channels,
#                  out_channels,
#                  num_layers,
#                  heads=1,
#                  groups=1):
#         super(Graph_Net, self).__init__()
#         self.hidden_channels = hidden_channels
#         self.lin1 = nn.Sequential(
#             nn.Linear(in_channels, 8),
#             nn.ReLU(),
#             nn.Linear(8,16),
#             nn.ReLU(),
#             nn.Linear(16,hidden_channels),
#         )
#         self.conv1 = GATConv(hidden_channels,hidden_channels//4,4,dropout=0.1)
#         self.conv2 = GATConv(hidden_channels,hidden_channels,1,concat=True,dropout=0.1)
#         # self.convs = torch.nn.ModuleList()
#         # for i in range(num_layers):
#         #     self.convs.append(
#         #         DNAConv(
#         #             hidden_channels, heads, groups, dropout=0.2, cached=False))
#         self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
#
#     # def reset_parameters(self):
#     #     self.lin1.reset_parameters()
#     #     for conv in self.convs:
#     #         conv.reset_parameters()
#     #     self.lin2.reset_parameters()
#
#     def forward(self, x,edge_index,batch):
#         b,c = x.size()
#         # edge_index = torch.stack([torch.zeros(b,dtype=torch.long),torch.arange(b,dtype=torch.long)]).cuda()
#         x = F.relu(self.lin1(x))
#         # x = F.dropout(x, p=0.1, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = global_mean_pool(x,batch)
#         x = self.lin2(x)
#
#         return x

class Cnn_Module(nn.Module):
    def __init__(self,dim):
        super(Cnn_Module,self).__init__()
        self.mode = system_configs.mode
        if 'm' in self.mode:
            self.move_line1 = nn.Sequential(
                nn.Linear(4,16),
                nn.Linear(16,32),
            )
            self.move_lstm = nn.LSTM(
                input_size=32,
                hidden_size=256,
                num_layers=3,
                batch_first=True
            )
        dims = {
            's':256,
            'm':256*2,
            'ms':256*3
        }[self.mode]
        self.out_line = nn.Sequential(
            nn.Linear(dims,256),
            nn.Dropout(p=0.3),
            nn.Linear(256,64),
            nn.Dropout(p=0.3),
            nn.Linear(64,4),
        )

        if 's' in self.mode:
            self.space = nn.Sequential(
                nn.Linear(16,128),
                nn.Linear(128,256),
            )

            self.space_lstm = nn.LSTM(
                input_size=256,
                hidden_size=256,
                num_layers=3,
                batch_first=True
            )
    def forward(self, x):
        if 'm' in self.mode:
            x2 = torch.stack([self.move_line1(tx.cuda()) for tx in x[1]])
            x2_out ,_=self.move_lstm(x2)
            # x2 = self.move_line3(torch.cat([x2_out[:, -1, :],self.move_line2(x2[:,-1])],dim=1))
            x2 = x2_out[:, -1, :]
            x2 = torch.cat([x2,x2_out[:, -2, :]],dim=1)

        if self.mode =='s':
            # x3 = torch.stack([self.space(torch.clamp(tx.cuda(),min=-2,max=2)) for tx in x[2]])
            x3 = torch.stack([self.space(tx.cuda()) for tx in x[2]])
            x3_out, _ = self.space_lstm(x3)
            x = x3_out[:, -1, :]
        elif  self.mode=='ms':
            x3 = torch.stack([self.space(tx.cuda()) for tx in x[2]])
            x3_out ,_=self.space_lstm(x3)
            x3 = x3_out[:, -1, :]
            x = torch.cat([x3,x2],dim=1)
        else:
            x = x2
        x = self.out_line(x)
        return x[:,:2],x[:,2:]



class AEloss(nn.Module):
    def __init__(self,beta = 1.5):
        super(AEloss,self).__init__()
        self.loss=nn.CrossEntropyLoss()
    # def forward(self, preds, gt):
    #     b,_,w,h = preds.size()
    #     gt = gt[0]
    #     miss_pairs = torch.abs(gt-preds)
    #     loss = miss_pairs/(b*w*h)
    #     return loss
    def forward(self, preds, gt):
        cla = gt[0].long()
        loss1 = self.loss(preds[0],cla)
        pred = torch.argmax(preds[0],dim=1)
        correct = (pred == cla).sum().float()/pred.size()[0]

        beta=1
        diff = torch.abs(preds[1] - gt[1])
        test1 = diff*10
        test2 = gt[1]*10
        test3 = preds[1]*10
        loss2 = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta).mean()
        loss = loss1+loss2
        
        return loss,correct,loss1,loss2

