'''
Use saliency map for Temporal Attention, and filter the video content
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import random
import math


class AttentionLayer(nn.Module):
    '''
    Calcuate the attention weight according to the hidden state of LSTM
    and the CNN conv features
    '''
    def __init__(self, hidden_size, projected_size):
        '''
        hidden_size: hidden vector size of LSTM
        frame_embed_size: the input dimension of CNN feature
        projected_size: the projected dimension of LSTM state and CNN feature
        '''
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.projected_size = projected_size
        self.linear1 = nn.Linear(hidden_size, projected_size)
        self.linear2 = nn.Linear(projected_size, projected_size)
        self.linear3 = nn.Linear(projected_size, 1, bias=False)

    def forward(self, h, v):
        bsz, num_frames = v.size()[:2]
        e = []
        for i in range(num_frames):
            x = self.linear1(h) + self.linear2(v[:, i, :])
            x = F.tanh(x)
            x = self.linear3(x)
            e.append(x)
        e = torch.cat(e, 0)
        a = F.softmax(e.view(bsz, num_frames))
        return a