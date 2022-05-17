# -*- coding: utf-8 -*-
# Time : 2022/5/17 7:00 PM
# Author : sk-w
# Email : 15734082105@163.com
# File : dataset.py
# Project : AlexNet-Pytorch
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, dataDir, annotations_file, transform=None, target_transform=None):

        self.paths, self.labels = self.getData(annotations_file, dataDir)
        self.transform = transform
        self.target_transform = target_transform

    def getData(self, annotations_file, dataDir):

        with open(annotations_file, "r") as f:
            data = f.readlines()
            f.close()
        paths = []
        labels = []
        for i in data:
            path, label = i.strip().split(" ")
            paths.append(os.path.join(dataDir, path))
            labels.append(label)

        return paths, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_path = self.paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch.nn.functional as F

    imgTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    def labelTransform(x):
        one_hot = F.one_hot(torch.tensor(eval(x)), num_classes=10)
        return one_hot


    # train data
    trainData = CustomImageDataset("./dataset", "dataset/dataset/train10.txt", imgTransform, labelTransform)
    traindata = DataLoader(trainData, batch_size=4, shuffle=True)

    data = next(iter(trainData))

    # from torch.utils.data import  DataLoader
    # import matplotlib.pyplot as plt
    # import torchvision
    # import  numpy as np
    # import json
    #
    # plt.rcParams["font.family"] = 'Arial Unicode MS'
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # # no transform
    # dataset = CustomImageDataset("./dataset","./dataset/train10.txt")
    # traindata = DataLoader(dataset,batch_size=4,shuffle=True)
    #
    # labelsJson = "./dataset/garbage_classification.json"
    # f = open(labelsJson, 'r')
    # info_data = json.load(f)
    #
    # for imgs,labels in iter(traindata):
    #     plt.figure("sample")
    #     img = torchvision.utils.make_grid(imgs)
    #     imgNp = img.numpy()
    #     plt.imshow(np.transpose(imgNp, (1, 2, 0)))  # 将【3，32，128】-->【32,128,3】
    #     # plt.title(labels)
    #     title = ""
    #     for label in labels:
    #         title = title + "_" + info_data[label].split("_")[1]
    #     plt.title(title)
    #     plt.show()
