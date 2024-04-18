import numpy as np
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import  matplotlib.pyplot as plt
import os
from torch import nn
from torch.nn import init
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False, transform=transforms.ToTensor())
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
'''
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
'''
batch_size = 256
train_iter = data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)
'''
class MyModal(nn.Module):
    def __init__(self,input_size,outout_size):
        super().__init__()
        self.linear = nn.Linear(input_size,outout_size)
    def forward(self,x):
        y=self.linear(x)
        return y
net = MyModal()
'''
net = nn.Sequential(nn.Flatten(),nn.Linear(28*28,10))
loss = nn.CrossEntropyLoss()
num_epochs, lr = 100, 0.2
'''
o,p=iter(train_iter).next()
o1 = net(o)
o2 = o1.argmax(dim=1)
o3 = p==o2
o4 = o3.float().sum()
o5 = o4.item()
'''
def evaluate_accuracy(test_iter,net):
    acc_sum, n = 0, 0
    for X,y in test_iter:
        acc_sum+=(y==net(X).argmax(dim=1)).float().sum().item()
        n+=y.shape[0]
    return  acc_sum/n
print(evaluate_accuracy(test_iter, net))
def train(net,loss,lr,num_epochs,train_iter,test_iter):
    optimizer = torch.optim.SGD(net.parameters(), lr)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X),y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        l= loss(net(X),y)
        test_acc=evaluate_accuracy(test_iter,net)
        print(f'epoch {epoch + 1}, loss {l:f}, test_accuracy{test_acc:f}')
train(net,loss,lr,num_epochs,train_iter,test_iter)
X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])




