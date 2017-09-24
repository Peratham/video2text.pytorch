import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import random
import math
import os

import misc.cnn.resnet as resnet

def build_cnn(opt):
    opt.pre_ft = getattr(opt, 'pre_ft', 1)

    if opt.pre_ft == 0:
        if opt.cnn_model == 'resnet101':
            net = getattr(resnet, opt.cnn_model)()
            # if vars(opt).get('start_from', None) is None and vars(opt).get('cnn_weight', '') != '':
            if len(opt.start_from) == 0 and len(opt.cnn_weight) != 0:
                net.load_state_dict(torch.load(opt.cnn_weight))
            net = nn.Sequential( \
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4)
            if len(opt.start_from) != 0:
                net.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_id + '.model-cnn-best.pth')))
        elif opt.cnn_model == 'vgg16':
            net = getattr(models, opt.cnn_model)()

            if len(opt.start_from) == 0 and len(opt.cnn_weight) != 0:
                net.load_state_dict(torch.load(opt.cnn_weight))
                print("Load pretrained CNN model from " + opt.cnn_weight)

            new_classifier = nn.Sequential(*list(net.classifier.children())[:6])
            net.classifier = new_classifier

            #if vars(opt).get('start_from', None) is None and vars(opt).get('cnn_weight', '') != '':
            if len(opt.start_from) != 0:
                print("Load pretrained CNN model (from start folder) : " + opt.start_from)
                net.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_id + '.model-cnn-best.pth')))
        net.cuda()
    else:
        net = None

    return net

class EncoderCNN(nn.Module):

    def __init__(self, opt):
        super(EncoderCNN, self).__init__()
        self.vgg = models.vgg16()
        self.vgg.load_state_dict(torch.load(opt.cnn_weight))
        # Remove the last fc layer of VGG (maintain the preceding ReLU layer)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])

    def forward(self, images):
        return self.vgg(images)
