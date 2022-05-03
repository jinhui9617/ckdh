import torch.nn as nn
import torchvision
import torch
import numpy as np
import copy
import math

# alexnet
class AlexNetNoTop(nn.Module):
    def __init__(self):
        super(AlexNetNoTop, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=True)
        del alexnet.classifier[6]
        self.alexnet = alexnet

    def forward(self, x):
        output = self.alexnet(x)
        return output


# Text classification pre training model
class TextBackbone(nn.Module):
    def __init__(self, embedding_length, num_label):
        super(TextBackbone, self).__init__()
        self.num_label = num_label
        self.embedding_length = embedding_length

        self.transition = nn.Sequential(
            nn.Linear(self.embedding_length, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Linear(1024, self.num_label),
            nn.Sigmoid()
        )

    def forward(self, y):
        y = self.transition(y)
        y = self.cls(y)

        return y


# Image classification pre training model
class ImgBackbone(nn.Module):
    def __init__(self, num_label):
        super(ImgBackbone, self).__init__()
        self.num_label = num_label
        self.alexnet = AlexNetNoTop()
        self.transition = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
        )
        self.cls = nn.Sequential(
            nn.Linear(1024, self.num_label),
            nn.Sigmoid()
        )

    def forward(self, image):
        x = self.alexnet(image)
        fea = self.transition(x)
        label = self.cls(fea)

        return label


# image student network
class ImageNet(nn.Module):
    def __init__(self, code_length, num_label):
        super(ImageNet, self).__init__()
        self.code_length = code_length
        self.num_label = num_label
        self.alexnet = AlexNetNoTop()
        self.transition = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
        )
        self.dh = nn.Sequential(
            nn.Linear(1024, self.code_length),
            nn.Tanh(),
        )
        self.cls = nn.Sequential(
            nn.Linear(self.code_length, self.num_label),
            nn.Sigmoid()
        )

    def forward(self, image):
        x = self.alexnet(image)
        fea = self.transition(x)
        code = self.dh(fea)
        label = self.cls(code)

        return fea, code, label


# text student network
class TextNet(nn.Module):
    def __init__(self, code_length, embedding_length, num_label):
        super(TextNet, self).__init__()
        self.code_length = code_length
        self.num_label = num_label
        self.embedding_length = embedding_length

        self.transition = nn.Sequential(
            nn.Linear(self.embedding_length, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
        )
        self.dh = nn.Sequential(
            nn.Linear(1024, self.code_length),
            nn.Tanh()
        )
        self.cls = nn.Sequential(
            nn.Linear(self.code_length, self.num_label),
            nn.Sigmoid()
        )

    def forward(self, y):
        fea = self.transition(y)
        code = self.dh(fea)
        label = self.cls(code)

        return fea, code, label


# student hashing network
class CKDH(nn.Module):
    def __init__(self, code_length, embedding_length, num_label):
        super(CKDH, self).__init__()
        self.imgnet = ImageNet(code_length, num_label)
        self.textnet = TextNet(code_length, embedding_length, num_label)

    def forward(self, img, text):
        fea_v, code_v, label_v = self.imgnet(img)
        fea_t, code_t, label_t = self.textnet(text)
        return fea_v, fea_t, code_v, code_t, label_v, label_t






