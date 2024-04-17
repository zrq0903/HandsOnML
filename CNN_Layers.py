import torch
from torch import  nn
import torch.nn.functional as F
def corr2d(X,K,stride):
    h,w = K.shape
    Y=torch.zeros((X.shape[0]-h)//stride+1,(X.shape[1]-w)//stride+1)
    for i in range(0,Y.shape[0]):
        for j in range(0,Y.shape[1]):
            Y[i][j]=(X[stride*i:stride*i+h,stride*j:stride*j+w]*K).sum()
    return Y
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K,2))
class Conv2D(nn.Module):
    def __init__(self,shape,stride=1,padding=1):
        super(Conv2D,self).__init__()
        self.weight = nn.Parameter(torch.randn(shape))
        self.bias = nn.Parameter(torch.randn(1))
        self.stride = stride
        self.padding = padding
    def forward(self,x):
        if self.padding != 0:
            x = F.pad(x,(1,1,1,1),mode='constant',value=0)
        return corr2d(x,self.weight,self.stride)+self.bias

Conv2dd = Conv2D(shape=(2,2),stride=2,padding=1)
print(Conv2dd(X))
print(Conv2dd.weight)
print(Conv2dd.bias)


