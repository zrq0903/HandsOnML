import numpy as np
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torch import nn
from torch.nn import init
from softmax_regression import evaluate_accuracy,train
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False, transform=transforms.ToTensor())
input_size ,hidden_layer,output_size=784,256,10
batch_size,lr, num_epoch = 256,0.1,5
train_iter = data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)
net = nn.Sequential(nn.Flatten(),nn.Linear(input_size,hidden_layer),nn.ReLU(),nn.Linear(hidden_layer,output_size))
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
loss = nn.CrossEntropyLoss()
print(evaluate_accuracy(test_iter, net))
#optimizer= torch.optim.SGD(net.parameters(),lr=lr)
train(net,loss,lr,num_epoch,train_iter,test_iter)



