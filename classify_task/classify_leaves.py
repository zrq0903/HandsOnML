from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torch import nn
from PIL import Image
import pandas as pd
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
from functions import eva_accu,train
'''
img_dir='./HandsOnDL/leave_images/0.jpg'
img=Image.open(img_dir)
print(img.size)
'''
train_csv=pd.read_csv('./HandsOnDL/classify_task/train.csv')
'''
class_to_num = train_csv.label.unique()
dict={}
for i in range(len(class_to_num)):
    dict[class_to_num[i]]=i
train_csv['class_num']=train_csv['label'].apply(lambda x:dict[x])
'''
leaves_labels = list(set(train_csv['label']))
lables_num=len(leaves_labels)   #len=176
class_to_num = dict(zip(leaves_labels,range(lables_num)))
num_to_class = dict(zip(range(lables_num),leaves_labels))
class LeavesDataset(Dataset):
    def __init__(self,root_dir,csv_path,mode='train',resize_height=224,resize_width=224,t_v_ratio=0.1):
        self.root_dir = root_dir
        self.mode = mode
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.t_v_ratio = t_v_ratio
        self.data_info = pd.read_csv(csv_path,header=0)
        self.data_len=len(self.data_info.index)
        self.train_len = int(self.data_len*(1-self.t_v_ratio))
        if mode == 'train':
            self.images = self.data_info.iloc[0:self.train_len,0].values
            self.labels = self.data_info.iloc[0:self.train_len,1].values
        if mode == 'valid':
            self.images = self.data_info.iloc[self.train_len:,0].values
            self.labels = self.data_info.iloc[self.train_len:,1].values
        if mode == 'test':
            self.images = self.data_info.iloc[:,0].values
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        img_name = self.images[item]
        img_item_path = os.path.join(self.root_dir,img_name)
        img = Image.open(img_item_path)
        if self.mode == 'train':
            dataset_trans = transforms.Compose([transforms.Resize((self.resize_height,self.resize_width)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.ToTensor()
                                      ])
        else:
            dataset_trans = transforms.Compose([transforms.Resize((self.resize_height,self.resize_width)),
                                        transforms.ToTensor()
                                                ])
        img = dataset_trans(img)
        if self.mode == 'test':
            return img
        else:
            label = class_to_num[self.labels[item]]
            return img,label

root_dir = './'
train_path = './HandsOnDL/classify_task/train.csv'
test_path = './HandsOnDL/classify_task/test.csv'

train_dataset = LeavesDataset(root_dir,train_path)
valid_dataset = LeavesDataset(root_dir,train_path,mode='valid')
test_dataset = LeavesDataset(root_dir,test_path,mode='test')
'''
img,label = train_dataset[0]
print(img.shape)
'''
batch_size = 256
train_iter = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
valid_iter = data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
#test_iter = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
net = models.resnet18(pretrained=True)
new_net = net
new_net.fc = nn.Linear(512,176)
nn.init.xavier_uniform_(new_net.fc.weight)
lr ,num_epochs = 0.001, 15
optimizer = torch.optim.Adam(new_net.parameters(),lr)
train(new_net,train_iter,valid_iter,num_epochs,optimizer)
print('done')
new_net.eval()
test_iter = data.DataLoader(test_dataset,batch_size=1,shuffle=False)
#a=next(iter(test_iter))
ans_num,ans_class = [],[]
with torch.no_grad():
    for image in test_iter:
        Y = new_net(image).argmax(dim=1).item()
        ans_num.append(Y)
for i in ans_num:
    ans_class.append(num_to_class[i])
test_csv=pd.read_csv('./HandsOnDL/classify_task/test.csv')
test_csv['label']=pd.Series(ans_class)
submission = pd.concat([test_csv['image'], test_csv['label']], axis=1)
submission.to_csv('./HandsOnDL/classify_task/submission.csv', index=False)

