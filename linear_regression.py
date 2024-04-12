import numpy as np
import torch
from torch.utils import data
x=torch.rand(1000,1)
y=x*10
z=torch.normal(mean=0,std=1,size=(1000,1))
y=y+z
bias=torch.ones(1000,1)
y=y+bias*5
data_iter=data.TensorDataset(x,y)
from torch import nn
linear=nn.Sequential(nn.Linear(1,1))
#linear[0].weight.data.normal_(0, 0.01)
#linear[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(linear.parameters(), lr=0.2)
batch_size = 500
data_loader = data.DataLoader(data_iter, batch_size)
print(next(iter(data_loader)))
epoch_num = 200
for epoch in range(epoch_num):
    for a, b in data_loader:
        l = loss(linear(a), b)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(linear(x), y)
    print(f'epoch {epoch + 1}, loss {l:f}')
w = linear[0].weight.data
print(w)
b = linear[0].bias.data
print(b)



