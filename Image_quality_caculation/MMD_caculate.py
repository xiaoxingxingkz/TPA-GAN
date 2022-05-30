
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

from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def guassian_kernel(source, target, kernel_mul=3.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    #求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    n_samples = int(source.size()[0]) + int(target.size()[0]) 
    
    #将source, target按列方向合并
    total = torch.cat([source, target], dim=0) 
    
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    #求任意两个数据之间的和，
    #得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2) 

    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值
    #（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss #因为一般都是n==m，所以L矩阵一般不加入计算


# caculat MMD of 2 groups 
sd = []
result_mmd = []
j = 1
if j == 1:
    sum_mmd = 0.0
    path_dir = './Generated_and_Real_PET_Results' 
    vector = []
    label = os.listdir(path_dir)
    for j in label:
        vector.append(j[0:12])
    vector = sorted(set(vector),key=vector.index)

    total_num = 329 #  Number of samples in Testset.
    for s in vector:
        mmd_v = 0.0
        img_pet_name = os.path.join(path_dir, str(s) + '_pet.mat')
        img_pet = sio.loadmat(img_pet_name) 
        imga = img_pet['data']

        img_fake_name = os.path.join(path_dir, str(s) + '_fake.mat')
        img_fake = sio.loadmat(img_fake_name) 
        imgb = img_fake['data']


        for i in range(76):
            X = torch.Tensor(imga[:, :, i])
            Y = torch.Tensor(imgb[:, :, i])
            X,Y = Variable(X), Variable(Y)
            mmd_v += mmd_rbf(X, Y)
        mmd_v = mmd_v/76.0
        sd.append(mmd_v)

        sum_mmd += mmd_v
    mean_mmd = sum_mmd/total_num       
    result_mmd.append(mean_mmd)
result_mmd = np.array(result_mmd)
print(result_mmd) # This is mean value！

result_sd = np.array(sd)
arr_mean = np.mean(result_sd)
arr_std = np.std(result_sd,ddof=1)
print("Mean Value:%f" % arr_mean)
print("Standard Deviation:%f" % arr_std)