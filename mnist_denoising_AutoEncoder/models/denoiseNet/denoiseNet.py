from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from ..base import BaseModel
import numpy as np
import os
import sys
from xml.etree import ElementTree

class DenoiseNet(BaseModel):
  def __init__(self, pretrained=True):
    super(DenoiseNet,self).__init__()
    
    ## Encoder layers
    self.features = nn.Sequential(OrderedDict([
    ('layer1', nn.Sequential(OrderedDict([
    	('conv1_1', nn.Conv2d(1,64,3,padding=1)),
    	('leaky1_1', nn.LeakyReLU(0.1, inplace=True)),
    	('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2))
    ]))),
    ('layer2', nn.Sequential(OrderedDict([
    	('conv2_1', nn.Conv2d(64,32,3,padding=1)),
    	('leaky2_1', nn.LeakyReLU(0.1, inplace=True)),
    	('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
    ]))),
    ('layer3', nn.Sequential(OrderedDict([
    	('conv3_1', nn.Conv2d(32,16,3,padding=1)),
    	('leaky3_1', nn.LeakyReLU(0.1, inplace=True)),
    	('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2))
    ]))),
    ('layer4', nn.Sequential(OrderedDict([
    	('conv4_1', nn.Conv2d(16,8,3,padding=1)),
    	('leaky4_1', nn.LeakyReLU(0.1, inplace=True)),
    	('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2))
    ]))),
    ('layer5', nn.Sequential(OrderedDict([
    	('tconv5_1',  nn.ConvTranspose2d(8,8,3,stride=2)),
    	('leaky5_1', nn.LeakyReLU(0.1, inplace=True))
    ]))),
    ('layer6', nn.Sequential(OrderedDict([
    	('tconv6_1', nn.ConvTranspose2d(8,16,3,stride=2)),
    	('leaky6_1', nn.LeakyReLU(0.1, inplace=True))
    ]))),
    ('layer7', nn.Sequential(OrderedDict([
    	('tconv7_1', nn.ConvTranspose2d(16,32,2,stride=2)),
    	('leaky7_1', nn.LeakyReLU(0.1, inplace=True))
    ]))),
    ('layer8', nn.Sequential(OrderedDict([
    	('tconv8_1', nn.ConvTranspose2d(32,64,2,stride=2)),
    	('leaky8_1', nn.LeakyReLU(0.1, inplace=True))
    ])))
    ]))
    
    self.classifier = nn.Sequential(OrderedDict([
    ('classifier', nn.Sequential(OrderedDict([
    ('conv8_1', nn.Conv2d(64,1,3,padding=1)),
    ('sigmoid', nn.Sigmoid())
    ])))
    ]))
  
    if pretrained:
            self.load_weight()
            print('Model is loaded')
            
  def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out
    
  def load_weight(self):
        weight_file = './models/denoiseNet/pytorch_weights/DenoiseNet_weights.pth'
        assert len(torch.load(weight_file).keys()) == len(self.state_dict().keys())
        dic = {}
        for now_keys, values in zip(self.state_dict().keys(), torch.load(weight_file).values()):
            dic[now_keys]=values
        self.load_state_dict(dic)
        print('Weights are loaded!')
    
   
