#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F

import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # try:
        outputs = torch.sigmoid(self.linear(x))
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs

class lenet5(nn.Module):  #继承nn.Module类，该类能够继承父类已经存在的属性和方法
    def __init__(self):
        super(lenet5,self).__init__()  #表示初始化时调用父类nn.Module的构造函数，使得子类能够继承父类的属性(变量)和方法（函数）
        #接下来自定义lenet5类的一些函数
        #定义卷积单元
        self.conv_unit= nn.Sequential(
            nn.Conv2d(
                3,  #输入通道数
                6, #输出通道数
                5, #5*5卷积核大小
                1  ,   #步进精度是5
                0
                     ),        #卷积层
            nn.AvgPool2d(
                    2, #窗口2*2
                    2,     #步进
                    0    #填充0
                    ),
            nn.Conv2d(6,16,5,1,0),
            nn.AvgPool2d(2,2,0)
        )
 
        self.fc_unit= nn.Sequential(
            nn.Linear(400,120), #全连接层输入400是因为卷积层输出为400
            nn.ReLU(),  #激活函数
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    def forward(self,x):   #调用类lenet5时，对输入x进行前向传播
            batchsz= x.size(0) #获取图片数量
            x=self.conv_unit(x) #先经过卷积层  设计网络时经过测试，[batch,3,32,32]输入 输出[batch,16,5,5]
            x=x.view(batchsz,16*5*5)  #将卷积输出转换为[batchsz,16*5*5]用于接下来输入全连接层
 
            logits= self.fc_unit(x) #输入全连接输出维度为[batch,10]
            return logits


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class CNNMnistHalf(nn.Module):
    def __init__(self):
        super(CNNMnistHalf, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),  # 将通道数减半
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2),  # 将通道数减半
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 16, 10)  # 调整全连接层的输入特征数量

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class CNNMnistQuarter(nn.Module):
    def __init__(self):
        super(CNNMnistQuarter, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, padding=2),  # 通道数减至4
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, padding=2),  # 通道数减至8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 8, 10)  # 调整全连接层的输入特征数量

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x

class CNNCifarHalf(nn.Module):
    def __init__(self):
        super(CNNCifarHalf, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # 将通道数减半
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)  # 将通道数减半
        self.conv3 = nn.Conv2d(64, 64, 3)  # 将通道数减半
        self.fc1 = nn.Linear(64 * 4 * 4, 10)  # 调整全连接层的输入特征数量

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)  # 调整视图
        x = self.fc1(x)
        return x
    
class CNNCifarQuarter(nn.Module):
    def __init__(self):
        super(CNNCifarQuarter, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # 将通道数减至32
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)  # 将通道数减至32
        self.conv3 = nn.Conv2d(32, 32, 3)  # 将通道数减至32
        self.fc1 = nn.Linear(32 * 4 * 4, 10)  # 调整全连接层的输入特征数量

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 4 * 4)  # 调整视图
        x = self.fc1(x)
        return x

class CNNCifar100(nn.Module):
    def __init__(self):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x
    
class CNNCifar100Half(nn.Module):
    def __init__(self):
        super(CNNCifar100Half, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)  # 将通道数减半
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)  # 将通道数减半
        self.conv3 = nn.Conv2d(128, 64, 3)  # 将通道数减半
        self.fc1 = nn.Linear(64 * 4 * 4, 100)  # 调整全连接层的输入特征数量

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)  # 调整视图
        x = self.fc1(x)
        return x

class CNNCifar100Quarter(nn.Module):
    def __init__(self):
        super(CNNCifar100Quarter, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # 将通道数减至64
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)  # 将通道数减至64
        self.conv3 = nn.Conv2d(64, 32, 3)  # 将通道数减至32
        self.fc1 = nn.Linear(32 * 4 * 4, 100)  # 调整全连接层的输入特征数量

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 4 * 4)  # 调整视图
        x = self.fc1(x)
        return x

class CNNCifar2(nn.Module):  # 重新搭建CNN
    def __init__(self):
        super(CNNCifar2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
