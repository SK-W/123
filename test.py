# -*- coding: utf-8 -*-
# Time : 2022/5/17 8:23 PM
# Author : sk-w
# Email : 15734082105@163.com
# File : test.py
# Project : AlexNet-Pytorch
import json

# JSON到字典转化
labelsJson = "./dataset/garbage_classification.json"
f = open(labelsJson, 'r')
info_data = json.load(f)
print(info_data['0'])
# 显示数据类型
print(type(info_data))
