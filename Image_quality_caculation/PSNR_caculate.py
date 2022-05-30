import numpy as np
import torch
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

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

sd = []
result_psnr = []
j = 1
if j == 1:
    sum_psnr = 0.0
    path_dir = './Generated_and_Real_PET_Results' 

    vector = []
    label = os.listdir(path_dir)
    for j in label:
        vector.append(j[0:12])
    vector = sorted(set(vector),key=vector.index)

    total_num = 329 #  Number of samples in Testset.
    for i in vector:
        psnr_v = 0.0
        img_pet_name = os.path.join(path_dir, str(i) + '_pet.mat')
        img_pet = sio.loadmat(img_pet_name) 
        imga = img_pet['data']
    
        img_fake_name = os.path.join(path_dir, str(i) + '_fake.mat')
        img_fake = sio.loadmat(img_fake_name) 
        imgb = img_fake['data']

        for s in range(76):
            psnr_v += compare_psnr(imga[:, :, s], imgb[:, :, s], data_range=255)
        psnr_v = psnr_v.item()/76.0
        sd.append(psnr_v)
        
        sum_psnr += psnr_v
    mean_psnr = sum_psnr/total_num
    result_psnr.append(mean_psnr)
result_psnr = np.array(result_psnr)
print(result_psnr) # This is mean value！

result_sd = np.array(sd)
arr_mean = np.mean(result_sd)
arr_std = np.std(result_sd,ddof=1)
print("Mean Value:%f" % arr_mean)
print("Standard Deviation:%f" % arr_std)