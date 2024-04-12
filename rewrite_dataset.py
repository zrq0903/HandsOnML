from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
class NewDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)
root_dir='./HandsOnDL/hymenoptera_data/train'
label_dir='ants'
ants_dataset= NewDataset(root_dir,label_dir)
bees_label_dir='bees'
bees_dataset=NewDataset(root_dir,bees_label_dir)
train_dataset= ants_dataset + bees_dataset
img1,__ = train_dataset[123]
img2,__ = train_dataset[124]
img1.show()
img2.show()