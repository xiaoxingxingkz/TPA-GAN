import numpy as np
import torch
import os
from skimage.metrics import structural_similarity as compare_ssim

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



result_ssim = []
sd = []
j = 1
if j == 1:
    sum_ssim = 0
    path_dir = './Generated_and_Real_PET_Results' 
    vector = []
    label = os.listdir(path_dir)
    for j in label:
        vector.append(j[0:12])
    vector = sorted(set(vector),key=vector.index)

    total_num = 329 #  Number of samples in Testset.
    for i in vector:
        ssim_v = 0.0
        img_pet_name = os.path.join(path_dir, str(i) + '_pet.mat')
        img_pet = sio.loadmat(img_pet_name) 
        imga = img_pet['data']

        img_fake_name = os.path.join(path_dir, str(i) + '_fake.mat')
        img_fake = sio.loadmat(img_fake_name) 
        imgb = img_fake['data']
        
        for i in range(76):
            ssim_v += compare_ssim(imga[:, :, i], imgb[:, :, i], win_size=11, gradient=False, data_range=255, multichannel=False, gaussian_weights=False, full=False, dynamic_range=None)
        ssim_v = ssim_v.item()/76.0
        sd.append(ssim_v)

        sum_ssim += ssim_v
    mean_ssim = sum_ssim/total_num
    result_ssim.append(mean_ssim)
result_ssim = np.array(result_ssim)
print(result_ssim) # This is mean value！

result_sd = np.array(sd)
arr_mean = np.mean(result_sd)
arr_std = np.std(result_sd,ddof=1)
print("Mean Value:%f" % arr_mean)
print("Standard Deviation:%f" % arr_std)