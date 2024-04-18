import numpy as np
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import  matplotlib.pyplot as plt
import os
from torch import nn
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False, transform=transforms.ToTensor())
batch_size = 256
train_iter = data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2)
        )# 16*5*5
        self.flatten = nn.Flatten()
        self.fc_layer=nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )
    def forward(self,x):
        feature = self.conv_layer(x)
        return self.fc_layer(self.flatten(feature))
net=AlexNet()
print(net)

def eva_accu(data_iter,net):
    accu_sum , n =0.0,0
    for X,y in data_iter:
        net.eval()
        accu_sum+=(y==net(X).argmax(dim=1)).float().sum().item()
        net.train()
        n+=y.shape[0]
    return  accu_sum/n

def init_weight(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
print(eva_accu(test_iter,net))
net.apply(init_weight)
print(eva_accu(test_iter,net))
lr ,num_epochs = 0.001, 10
optimizer = torch.optim.Adam(net.parameters(),lr)
a=[]
b=[]
c=[]
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
        a.append(train_l_sum/batch_num)
        b.append(train_accu_sum/n)
        c.append(test_accu)
        print('epooch {},train loss {:.3f},train accu {:.3f},test accu{:.3f}'.format(epoch+1,train_l_sum/batch_num,train_accu_sum/n,test_accu))
train(net,train_iter,test_iter,num_epochs,optimizer)
plt.plot(a,label = 'train_loss')
plt.plot(b,label = 'train_acc')
plt.plot(c,label = 'test_acc')
plt.legend()
plt.xlabel('epoches')
plt.show()

