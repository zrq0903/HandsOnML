import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch import nn
from  torch.utils import data
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
T = 1000
time = torch.arange(1,T+1,dtype=torch.float32)
x = torch.sin(0.01*time)+torch.normal(0,0.2,size=(T,))
tau = 4
features = torch.zeros(size=(T-tau,tau))
for i in range(tau):
    features[:,i]= x[i:i+T-tau]
labels = x[tau:].reshape(-1,1)
batch_size= 16
dataset = data.TensorDataset(features,labels)
train_iter = data.DataLoader(dataset,batch_size)
def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
net = nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
net.apply(init_weight)
def train(net,train_iter,loss,num_epochs,lr):
    optimizer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X),y)
            l.backward()
            optimizer.step()
        print('epoch = {}, loss = {:.3f}'.format(epoch+1,l))
loss=nn.MSELoss()
train(net,train_iter,loss,5,0.01)
preds = net(features)
plt.plot(time,x.detach().numpy(),label='data')
plt.plot(time[tau:],preds.detach().numpy(),label='1-step preds')
plt.xlabel('time')
plt.ylabel('x')
plt.legend()
plt.show()




