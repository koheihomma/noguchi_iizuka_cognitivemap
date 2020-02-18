import h5py
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

root_path = "/Users/honmakohei/Google_Drive（koheihomma@sacral.c.u-tokyo.ac.jp）/noguchi_iiduka/3DEnvForGAN"
hdfpath="data_homma.h5"
data_size = 1003

def Resize_train(root_path):
    for angle in [0, 90, 180, 270]:
        for i in tqdm(range(1, data_size*100+1)):
            image = root_path+"/ScreenshotFolder_train/"+str(i).zfill(4)+" shot_"+str(angle)+".png"
            with open(image, 'rb')as f:
                data=Image.open(f)
                data=data.resize((8, 8))
                data.save(image)
    print("Resized!")

def Resize_test(root_path):
    for angle in [0, 90, 180, 270]:
        for i in tqdm(range(1, data_size*10+1)):
            image = root_path+"/ScreenshotFolder_test/"+str(i).zfill(4)+" shot_"+str(angle)+".png"
            with open(image, 'rb')as f:
                data=Image.open(f)
                data=data.resize((8, 8))
                data.save(image)
    print("Resized!")

def get_concat_h(im1, im2, im3, im4):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width + im4.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width + im2.width, 0))
    dst.paste(im4, (im1.width + im2.width + im3.width, 0))
    return dst

def Concat_train(path, images):
    for i in tqdm(range(1, data_size*100+1)):
        im_0 = path+str(i).zfill(4)+" shot_0.png"
        im_90 = path+str(i).zfill(4)+" shot_90.png"
        im_180 = path+str(i).zfill(4)+" shot_180.png"
        im_270 = path+str(i).zfill(4)+" shot_270.png"

        im_0 = Image.open(im_0)
        im_90 = Image.open(im_90)
        im_180 = Image.open(im_180)
        im_270 = Image.open(im_270)

        im_h =get_concat_h(im_0, im_90, im_180, im_270)
        im_h = np.asarray(im_h.convert('RGB'), dtype="float32") / 255

        im_h = im_h.reshape(1, 768)
        images.append(im_h)
    print("Concatenated!")

def Concat_test(path, images):
    for i in tqdm(range(1, data_size*10+1)):
        im_0 = path+str(i).zfill(4)+" shot_0.png"
        im_90 = path+str(i).zfill(4)+" shot_90.png"
        im_180 = path+str(i).zfill(4)+" shot_180.png"
        im_270 = path+str(i).zfill(4)+" shot_270.png"

        im_0 = Image.open(im_0)
        im_90 = Image.open(im_90)
        im_180 = Image.open(im_180)
        im_270 = Image.open(im_270)

        im_h =get_concat_h(im_0, im_90, im_180, im_270)
        im_h = np.asarray(im_h.convert('RGB'), dtype="float32") / 255

        im_h = im_h.reshape(1, 768)
        images.append(im_h)
    print("Concatenated!")

def MotionPositionSave(file_name):
    with open(file_name) as f:
        data_lines = f.read()
        data_lines = data_lines.replace("(", "")
        data_lines = data_lines.replace(")", "")
        data_lines = data_lines.replace(",", "")

    with open(file_name, mode="w") as f:
        f.write(data_lines)

def MotionPositionToNumpy(file_name):
    m=np.loadtxt(file_name, dtype="float32")
    return m



Resize_train(root_path)
Resize_test(root_path)
all_images_train = []
path_train=root_path+"/ScreenshotFolder_train/"
Concat_train(path_train, all_images_train)
all_images_train = np.array(all_images_train, dtype="float32")
all_images_train.reshape(100, data_size, 768)

all_images_test = []
path_test=root_path+"/ScreenshotFolder_test/"
Concat_test(path_test, all_images_test)
all_images_test = np.array(all_images_test, dtype="float32")
all_images_test.reshape(10, data_size, 768)

for mode in ["test", "train"]:
    file_name_m = root_path+"/Motion_"+mode+".txt"
    file_name_p = root_path+"/Position_"+mode+".txt"
    MotionPositionSave(file_name_m)
    MotionPositionSave(file_name_p)

all_motion_train = MotionPositionToNumpy(root_path+"/Motion_train.txt")
all_motion_test = MotionPositionToNumpy(root_path+"/Motion_test.txt")
all_position_train = MotionPositionToNumpy(root_path+"/Position_train.txt")
all_position_test = MotionPositionToNumpy(root_path+"/Position_test.txt")

print(all_motion_train.shape,all_position_train.shape,all_images_train.shape)

images_train = []
images_test = []
motion_train = []
motion_test = []
position_train = []
position_test = []

idx=0
idx2=0

for i in tqdm(range(100)):
    idx = 1003 * i
    idx2= 1003 * (i + 1)
    motion_train.append(all_motion_train[idx:idx2])
    position_train.append(all_position_train[idx:idx2])
    images_train.append(all_images_train[idx:idx2])

for i in tqdm(range(10)):
    idx = 1003 * i
    idx2= 1003 * (i + 1)
    motion_test.append(all_motion_test[idx:idx2])
    position_test.append(all_position_test[idx:idx2])
    images_test.append(all_images_test[idx:idx2])


motion_train=np.array(motion_train, dtype="float32")
position_train=np.array(position_train, dtype="float32")
images_train=np.array(images_train, dtype="float32")
motion_test=np.array(motion_test, dtype="float32")
position_test=np.array(position_test, dtype="float32")
images_test=np.array(images_test, dtype="float32")
#images = np.where(images<0.5,0.,1.) #binarization
images_train = images_train.reshape(100, data_size, 768)
images_test = images_test.reshape(10, data_size, 768)

print(motion_train.shape,position_train.shape,images_train.shape)



with h5py.File(hdfpath, 'w') as f:
    grp_m = f.create_group('motion')
    grp_m.create_dataset("train", data=motion_train)
    grp_m.create_dataset("test", data=motion_test)
    grp_p = f.create_group('position')
    grp_p.create_dataset("train", data=position_train)
    grp_p.create_dataset("test", data=position_test)
    grp_v = f.create_group('vision')
    grp_v.create_dataset("train", data=images_train)
    grp_v.create_dataset("test", data=images_test)

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
