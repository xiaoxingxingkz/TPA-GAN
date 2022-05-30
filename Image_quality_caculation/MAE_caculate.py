
import numpy as np
import torch
import os

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
import scipy.io as sio 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def mae(imageA, imageB):
	err = np.sum(abs(imageA.astype("float") - imageB.astype("float")))
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


# caculat MSE of 2 groups 
sd = []
result_mae = []
j = 1
if j == 1:
    sum_mae = 0.0
    path_dir = './Generated_and_Real_PET_Results' 
    vector = []
    label = os.listdir(path_dir)
    for j in label:
        vector.append(j[0:12])
    vector = sorted(set(vector),key=vector.index)

    total_num = 329 #  Number of samples in Testset.
    for s in vector:
        mae_v = 0.0
        img_pet_name = os.path.join(path_dir, str(s) + '_pet.mat')
        img_pet = sio.loadmat(img_pet_name) 
        imga = img_pet['data']

        img_fake_name = os.path.join(path_dir, str(s) + '_fake.mat')
        img_fake = sio.loadmat(img_fake_name) 
        imgb = img_fake['data']

        for i in range(76):
            mae_v += mae(imga[:, :, i], imgb[:, :, i])
        mae_v = mae_v/76.0
        sd.append(mae_v)

        sum_mae += mae_v
    mean_mae = sum_mae/total_num
    result_mae.append(mean_mae)
result_mae = np.array(result_mae)
print(result_mae) # This is mean value！

result_sd = np.array(sd)
arr_mean = np.mean(result_sd)
arr_std = np.std(result_sd,ddof=1)
print("Mean Value:%f" % arr_mean)
print("Standard Deviation:%f" % arr_std)