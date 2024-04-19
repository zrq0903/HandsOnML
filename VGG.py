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
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
dataset_transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False, transform=dataset_transform)
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False, transform=dataset_transform)
batch_size = 256
#feature, label = mnist_train[0]
#print(feature.shape, feature.dtype)
train_iter = data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)

print('loading completed')
def VGG_block(conv_num,in_size,out_size):
    block=[]
    for i in range(conv_num):
        if i==0:
            block.append(nn.Conv2d(in_channels=in_size,out_channels=out_size,kernel_size=3,padding=1))
        else:
            block.append(nn.Conv2d(in_channels=out_size,out_channels=out_size,kernel_size=3,padding=1))
    block.append(nn.MaxPool2d(2,stride=2))
    return nn.Sequential(*block)
block_info=((1,1,8),(1,8,16),(2,16,32),(2,32,64),(2,64,64))
def VGG(block_info,hidden_units,fc_size):
    net = nn.Sequential()
    for i,(conv_num,in_size,out_size) in enumerate(block_info):
        net.add_module(name='vgg_block'+str(i),module=VGG_block(conv_num,in_size,out_size))
    net.add_module(name='avg_pool',module=nn.AvgPool2d(1,stride=1))
    net.add_module(name='fc_layer',module=nn.Sequential(
        nn.Flatten(),
        nn.Linear(fc_size,hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units,hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units,10)
    ))
    return net
net = VGG(block_info,4096,64*7*7)

print(eva_accu(test_iter,net))
def init_weight(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weight)
print(eva_accu(test_iter,net))
lr ,num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(),lr)
a=[]
b=[]
c=[]
train(net,train_iter,test_iter,num_epochs,optimizer)
plt.plot(a,label = 'train_loss')
plt.plot(b,label = 'train_acc')
plt.plot(c,label = 'test_acc')
plt.legend()
plt.xlabel('epoches')
plt.show()
