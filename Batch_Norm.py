import numpy as np
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import os
from torch import nn
import matplotlib.pyplot as plt
from functions import eva_accu,train
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False, transform=transforms.ToTensor())
batch_size = 256
train_iter = data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)
def BN(is_training,X,moving_mean,moving_var,momentum,beta,gamma,eps):
    if not is_training:
        X_hat = (X - moving_mean)/torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:
            mean = X.mean(dim=(0,2,3),keepdim=True)
            var = ((X-mean)**2).mean(dim=(0,2,3),keepdim=True)
        X_hat = (X-mean)/torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1-momentum) * mean
        moving_var = momentum * moving_var + (1-momentum) * var
    y = gamma * X_hat + beta
    return y, moving_mean, moving_var
class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1,1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    def forward(self,x):
        y, moving_mean, moving_var = BN(self.training,x,self.moving_mean,self.moving_var,0.9,self.beta,self.gamma,1e-5)
        return y
net = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            BatchNorm(6, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16*4*4, 120),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
batch_size = 256
lr, num_epochs = 0.001, 15
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net,train_iter,test_iter,num_epochs,optimizer)


