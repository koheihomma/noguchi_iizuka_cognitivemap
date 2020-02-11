import sys

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pickle
import torch
from PIL import Image

#mode = sys.argv[1]
mode = "test"
idx = int(sys.argv[1])
#idx = 0

width = 8
height = 8
channel = 3
num_camera = 4

f_v = open("./vision_output.txt","rb")
f_h5 = h5py.File('data.h5', 'r')

vision = pickle.load(f_v)[idx].clone().detach()
data = f_h5['vision'][mode][()][idx]
vision, data = vision.reshape(-1, channel, width * num_camera, height), data.reshape(-1, channel, width * num_camera, height)
vision, data = vision.permute(0, 2, 3, 1), data.transpose(0, 2, 3, 1)


def get_img(t):
    img = vision[t]
    img = np.concatenate(np.split(img, 4, 0), 1)
    return img

def get_img_t(t):
    img_t = data[t+1]
    img_t = np.concatenate(np.split(img_t, 4, 0), 1)
    return img_t

ims=[]
idx=0
for i in [200, 220, 240, 260, 280]:
    im=get_img(i)
    im_t=get_img_t(i)
    for i in range(19):
        im=np.concatenate((im, get_img(10*idx+i+1)), axis=0)
        im_t=np.concatenate((im_t, get_img_t(10*idx+i+1)), axis=0)
    ims.append(im)
    ims.append(im_t)

    idx+=1


fig, ax = plt.subplots(1, 10, figsize=(20, 20), tight_layout=True)
for i in range(10):
    ax[i].imshow(ims[i])
    ax[i].set_yticks([])
    ax[i].set_xticks([])
    if i%2==0:
        ax[i].set_title('predict')
    else:
        ax[i].set_title('test')
plt.show()
