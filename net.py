# -*- coding: utf-8 -*-
# Time : 2022/5/17 12:28 PM
# Author : sk-w
# Email : 15734082105@163.com
# File : net.py
# Project : AlexNet-Pytorch

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        # get features networks
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=(1, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2)),
        )
        # get classifier
        self.classifier = nn.Sequential(
            # fc1
            nn.Dropout(p=0.5),
            nn.Linear(256 * 5 * 6, 2048),
            nn.ReLU(inplace=True),
            # fc2
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # fc3
            nn.Linear(2048, class_num)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    inputTensor = torch.rand(4, 3, 224, 224)
    net = AlexNet(10)
    print(net)
    output = net.forward(inputTensor)
