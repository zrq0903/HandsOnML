from PIL import Image
from matplotlib import pyplot as plt
import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
def bbox_to_rect(bbox,color): #(xmin, ymin, xmax, ymax)格式转换成matplotlib格式：((xmin,ymin), width, height)
    return plt.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],fill=False,edgecolor=color,linewidth=2)
'''
img=Image.open('./HandsOnDL/img/catdog.jpg')
fig = plt.imshow(img)
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
plt.show()
'''
def Read_banana_data(dir_name,is_train=True):
    csv_name = os.path.join(dir_name,'bananas_train' if is_train else 'bananas_val','label.csv')
    data = pd.read_csv(csv_name)
    data = data.set_index('img_name')
    images, targets = [],[]
    for name,target in data.iterrows():
        images.append(
            torchvision.io.read_image(os.path.join(dir_name,'bananas_train' if is_train else 'bananas_val','images',name))
        )
        targets.append(list(target))
    return  images,torch.tensor(targets).unsqueeze(1)/256
#images,targets = Read_banana_data('./banana-detection')

class BananaDataset(Dataset):
    def __init__(self,dir_name,is_train):
        self.images,self.lables = Read_banana_data(dir_name,is_train=is_train)
    def __getitem__(self, item):
        return self.images[item].float(),self.lables[item]
    def __len__(self):
        return len(self.images)
batch_size, edge_size = 32, 256
train_iter = DataLoader(BananaDataset('./banana-detection',is_train=True), batch_size, shuffle=True)
#val_iter = DataLoader(BananaDataset('./banana-detection',is_train=False), batch_size)
batch = next(iter(train_iter))
#aa,bb=batch[0].shape, batch[1].shape
def make_plot(imgs,rows,cols):
    _,axes = plt.subplots(nrows=rows,ncols=cols)
    for i in range(rows):
        for j in range(cols):
            axes[i][j].imshow(imgs[i*cols+j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes
def draw_bbox(axes,bboxes):
    for bbox in bboxes:
        rect = bbox_to_rect(bbox.numpy(),'w')
        axes.add_patch(rect)
imgs = batch[0][0:10].permute(0,2,3,1)/255
axes = make_plot(imgs,2,5)
for ax,bboxes in zip(axes[0],batch[1][0:5]):
    draw_bbox(ax,[bboxes[0][1:5]*256])
for ax,bboxes in zip(axes[1],batch[1][5:10]):
    draw_bbox(ax,[bboxes[0][1:5]*256])
plt.show()