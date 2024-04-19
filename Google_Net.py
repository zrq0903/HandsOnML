import numpy as np
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import  matplotlib.pyplot as plt
import os
from torch import nn
import matplotlib.pyplot as plt
from functions import eva_accu,train
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
dataset_transform=transforms.Compose([transforms.Resize(96),transforms.ToTensor()])
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False, transform=dataset_transform)
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False, transform=dataset_transform)
batch_size = 256
train_iter = data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)
print('loading completed')
class Inception(nn.Module):
    def __init__(self,in_cha,c1,c2,c3,c4):
        super().__init__()
        self.p1 = nn.Conv2d(in_channels=in_cha,out_channels=c1,kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels=in_cha,out_channels=c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0],out_channels=c2[1],kernel_size=3,padding=1)
        self.p3_1 = nn.Conv2d(in_channels=in_cha,out_channels=c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channels=in_cha,out_channels=c4,kernel_size=1)
    def forward(self,x):
        p1 = F.relu(self.p1(x))
        p2 = F.relu(self.p2_2(self.p2_1(x)))
        p3 = F.relu(self.p3_2(self.p3_1(x)))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2, 3))

def Googlenet():
    b1 = nn.Sequential(
        nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
        nn.ReLU(),
        nn.MaxPool2d(3,stride=2,padding=1)
                       )
    b2 = nn.Sequential(
        nn.Conv2d(64,64,kernel_size=1),
        nn.Conv2d(64,192,kernel_size=3,padding=1),
        nn.MaxPool2d(3,stride=2,padding=1)
                       )
    b3 = nn.Sequential(
        Inception(192,64,(96,128),(16,32),32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       GlobalAveragePooling())

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.Flatten(), nn.Linear(1024, 10))
    return net
net = Googlenet()
x=torch.rand(256,1,96,96)
print(net)
for layer in net:
    x=layer(x)
    print(x.shape)
lr ,num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(),lr)
train(net,train_iter,test_iter,num_epochs,optimizer)


