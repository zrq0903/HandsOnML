import numpy as np
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import os
from torch import nn
import matplotlib.pyplot as plt
from functions import eva_accu,train
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
dataset_transform=transforms.Compose([transforms.Resize(56),transforms.ToTensor()])
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False, transform=dataset_transform)
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False, transform=dataset_transform)
batch_size = 256
train_iter = data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)
print('loading completed')

class Residual_block(nn.Module):
    def __init__(self,in_cha,out_cha,use_1x1conv=False,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_cha,out_cha,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(out_cha,out_cha,kernel_size=3,padding=1,stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_cha,out_cha,kernel_size=1,stride=stride)
        else:
            self.conv3 = None
        self.bn1=nn.BatchNorm2d(out_cha)
        self.bn2=nn.BatchNorm2d(out_cha)
    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y+x)

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2, 3))

net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) #(shape,64,h/4,w/4)

def Resnet_block(in_cha,out_cha,block_num,is_first_block=False):
    blk=[]
    for i in range(block_num):
        if i==0 and not is_first_block:
            blk.append(Residual_block(in_cha,out_cha,use_1x1conv=True,stride=2))
        else:
            blk.append(Residual_block(out_cha,out_cha))
    return nn.Sequential(*blk)

net.add_module("resnet_block1",Resnet_block(64,64,2,is_first_block=True))
net.add_module("resnet_block2",Resnet_block(64,128,2))
net.add_module("global_avg_pool",GlobalAveragePooling())
net.add_module("fc",nn.Sequential(nn.Flatten(),nn.Linear(128,10)))
'''
x=torch.rand(64,1,112,112)
for name,layer in net.named_children():
    x=layer(x)
    print(name,x.shape)
'''
lr ,num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(),lr)
train(net,train_iter,test_iter,num_epochs,optimizer)


