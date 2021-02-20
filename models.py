# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    
    def __init__(self, args):
        super(TextCNN, self).__init__()
        V = args.num_embeddings
        D = args.D
        C = args.C
        Ci = args.Ci
        Co = args.Co
        Ks = args.Ks
        weight_matrix = args.weight
        static = args.static
        self.use_bn = args.use_bn
        
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(weight_matrix)
        if static:
            self.embed.weight.requires_grad=False
        
        self.bn2d = nn.BatchNorm2d(1,momentum=0.1)
        #can keep the size by set padding=(kernel_size-1)//2, if stride=1
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding=((K-1)//2,0)) for K in Ks])
        self.dropout = nn.Dropout(args.drop)
        self.fc = nn.Linear(len(Ks)*Co, C)

    def forward(self, x):
        x = self.embed(x)  # (N, L, D)
        
        # (N, Ci, L, D), insert a dimention of size one(in_channels Ci)
        x = x.unsqueeze(1)  
        
        # ModuleList can act as iterable, or be indexed using ints
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N, Co, L), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        
        #concatenate different feature from different kernel sizes
        x = torch.cat(x, 1)   

        x = self.dropout(x)  # (N, len(Ks)*Co)
    
        x = self.fc(x)  # (N, C)
        return x