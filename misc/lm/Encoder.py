from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import *
import misc.utils as utils

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.padd_idx = 0
        self.opt = opt
        self.frame_size = opt.frame_size
        self.hidden_size = opt.hidden_size
        self.num_frames = opt.num_frames
        self.projected_size = opt.projected_size
        self.drop_prob_lm = getattr(opt, 'drop_prob_lm', 0.8)
        # frame_embed is used to embed the frame feature to low-dim space
        self.vs_frame_embed = nn.Linear(self.frame_size, self.projected_size)
        self.vs_frame_drop = nn.Dropout(p=self.drop_prob_lm)
        self.vf_frame_embed = nn.Linear(self.frame_size, self.projected_size)
        self.vf_frame_drop = nn.Dropout(p=self.drop_prob_lm)
        self.frame_embed = nn.Linear(self.projected_size * 2, self.projected_size)
        self.frame_drop = nn.Dropout(p=self.drop_prob_lm)

        self._init_weights()

    def _init_weights(self):
        variance = math.sqrt(2.0 / (self.frame_size + self.projected_size))
        self.vs_frame_embed.weight.data.normal_(0.0, variance)
        self.vs_frame_embed.bias.data.zero_()
        self.vf_frame_embed.weight.data.normal_(0.0, variance)
        self.vf_frame_embed.bias.data.zero_()

    def forward(self, video_feats):
        # Encoding process!
        batch_size = len(video_feats)
        # vs is the feature of saliency region of each frame
        vs = video_feats[:, :, :self.frame_size].contiguous()
        vs = vs.view(-1, self.frame_size)
        vs = self.vs_frame_embed(vs)
        vs = self.vs_frame_drop(vs)
        vs_ = vs.view(batch_size, self.num_frames, -1)
        # vf if the full feature of each frame
        #vf = video_feats[:, :, self.frame_size:].contiguous()
        vf = video_feats[:, :, :self.frame_size].contiguous()
        vf = vf.view(-1, self.frame_size)
        vf = self.vf_frame_embed(vf)
        vf = self.vf_frame_drop(vf)
        # vf_ = vf_.view(batch_size, self.num_frames, -1)
        # vr is the residual between full frame feature and saliency region feature
        vr = vf - vs
        # v is the concatenated feature of vs, and vr
        v = torch.cat([vs, vr], 1)
        v = self.frame_embed(v)
        v = v.view(batch_size, self.num_frames, -1)
        return v, vs, vs_
