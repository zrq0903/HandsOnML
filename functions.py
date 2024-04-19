import numpy as np
import torch
from torch.utils import data
import torchvision
import os
from torch import nn
def eva_accu(data_iter,net):
    accu_sum,n =0.0,0
    for X,y in data_iter:
        net.eval()
        accu_sum+=(y==net(X).argmax(dim=1)).float().sum().item()
        net.train()
        n+=y.shape[0]
    return  accu_sum/n
def train(net,train_iter,test_iter,num_epochs,optimizer):
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        n , train_accu_sum , train_l_sum, batch_num = 0 ,0.0 ,0.0,0
        for X,y in train_iter:
            y_pred = net(X)
            l = loss(y_pred,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n+=y.shape[0]
            batch_num+=1
            train_l_sum+=l.item()
            train_accu_sum += (y==y_pred.argmax(dim=1)).float().sum().item()
        test_accu = eva_accu(test_iter,net)
        print('epooch {},train loss {:.3f},train accu {:.3f},test accu{:.3f}'.format(epoch+1,train_l_sum/batch_num,train_accu_sum/n,test_accu))