import h5py
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

root_path = "/Users/honmakohei/Google_Drive（koheihomma@sacral.c.u-tokyo.ac.jp）/noguchi_iiduka/3DEnvForGAN"

data_numper = 11
def Resize(root_path, seed):
    for angle in [0, 90, 180, 270]:
        for i in range(2, 1004):
            image = root_path+"/ScreenshotFolder_"+str(seed)+"/"+str(i).zfill(4)+" shot_"+str(angle)+".png"
            with open(image, 'rb')as f:
                data=Image.open(f)
                #data=data.convert('RGB')
                data=data.resize((8, 8))
                data.save(image)
    print("Resized!")

for seed in range(data_numper):
    Resize(root_path, seed)

def get_concat_h(im1, im2, im3, im4):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width + im4.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width + im2.width, 0))
    dst.paste(im4, (im1.width + im2.width + im3.width, 0))
    return dst

def Concat(path, images):
    for i in range(2,1004):
        im_0 = path+str(i).zfill(4)+" shot_0.png"
        im_90 = path+str(i).zfill(4)+" shot_90.png"
        im_180 = path+str(i).zfill(4)+" shot_180.png"
        im_270 = path+str(i).zfill(4)+" shot_270.png"

        im_0 = Image.open(im_0)
        im_90 = Image.open(im_90)
        im_180 = Image.open(im_180)
        im_270 = Image.open(im_270)

        im_h =get_concat_h(im_0, im_90, im_180, im_270)
        #im_h = np.array(im_h)
        im_h = np.asarray(im_h.convert('RGB')) / 255
        im_h = im_h.reshape(1, 768)
        images.append(im_h)
    print("Concatenated!")

images = []
for i in range(data_numper):
    images_sublist=[]
    path = root_path+"/ScreenshotFolder_"+str(i)+"/"
    Concat(path, images_sublist)
    images.append(images_sublist)

images = np.array(images)
images.reshape((data_numper, 1002, 768))


def MotionPositionSave(file_name):
    with open(file_name) as f:
        data_lines = f.read()
        data_lines = data_lines.replace("(", "")
        data_lines = data_lines.replace(")", "")
        data_lines = data_lines.replace(",", "")

    with open(file_name, mode="w") as f:
        f.write(data_lines)

for i in range(data_numper):
    file_name_m = root_path+'/Motion'+str(i)+'.txt'
    file_name_p = root_path+'/Position'+str(i)+'.txt'
    MotionPositionSave(file_name_m)
    MotionPositionSave(file_name_p)

def MotionPositionToNumpy(file_name):
    m=np.loadtxt(file_name)
    return m

motion=[]
position=[]
for i in range(data_numper):
    file_name_m = root_path+'/Motion'+str(i)+'.txt'
    file_name_p = root_path+'/Position'+str(i)+'.txt'
    m = MotionPositionToNumpy(file_name_m)
    p = MotionPositionToNumpy(file_name_p)
    motion.append(m[:1002])
    position.append(p[:1002])

motion=np.array(motion)
position=np.array(position)

print(motion.shape, position.shape,images.shape)

with h5py.File('Sample.h5', 'w') as f:
    grp_m = f.create_group('motion')
    #assert m.shape==(1001, 2)
    grp_m.create_dataset('train', data=motion)
    grp_p = f.create_group('position')
    grp_p.create_dataset('train', data=position)
    grp_v = f.create_group('vision')
    grp_v.create_dataset('train', data=images)

hdfpath="/Users/honmakohei/Google_Drive（koheihomma@sacral.c.u-tokyo.ac.jp）/noguchi_iiduka/dataset/Sample.h5"
def PrintOnlyDataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name)

with h5py.File(hdfpath,'r') as f:
    f.visititems(PrintOnlyDataset)
    print(f["position/train"][0][0])
    im=f["vision/train"][0][0]
    im=im.reshape(8, 32, 3)
    plt.imshow(im)
    plt.show()
