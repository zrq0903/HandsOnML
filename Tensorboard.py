from tensorboardX import SummaryWriter
writer=SummaryWriter("logs")
'''
for i in range(10):
    writer.add_scalar('y=x',i,i)
writer.close()
'''

import numpy as np
from PIL import Image
img_PIL=Image.open('C:/Users/zxczr/Desktop/passport.jpg')
img_array=np.array(img_PIL)
writer.add_image('passport',img_array,1,dataformats='HWC')
writer.close()

