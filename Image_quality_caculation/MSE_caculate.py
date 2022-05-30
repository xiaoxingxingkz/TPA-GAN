
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


def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


sd = []
result_mse = []
j = 1
if j == 1:
    sum_mse = 0.0
    path_dir = './Generated_and_Real_PET_Results' 
    vector = []
    label = os.listdir(path_dir)
    for j in label:
        vector.append(j[0:12])
    vector = sorted(set(vector),key=vector.index)

    total_num = 329  #  Number of samples in Testset.
    for s in vector:
        mse_v = 0.0
        img_pet_name = os.path.join(path_dir, str(s) + '_pet.mat')
        img_pet = sio.loadmat(img_pet_name) 
        imga = img_pet['data']

        img_fake_name = os.path.join(path_dir, str(s) + '_fake.mat')
        img_fake = sio.loadmat(img_fake_name) 
        imgb = img_fake['data']

        for i in range(76):
            mse_v += mse(imga[:, :, i], imgb[:, :, i])
        mse_v = mse_v/76.0
        sd.append(mse_v)

        sum_mse += mse_v
    mean_mse = sum_mse/total_num
    result_mse.append(mean_mse)
result_mse = np.array(result_mse)
print(result_mse) # This is mean value！

result_sd = np.array(sd)
arr_mean = np.mean(result_sd)
arr_std = np.std(result_sd,ddof=1)
print("Mean Value:%f" % arr_mean)
print("Standard Deviation:%f" % arr_std)
