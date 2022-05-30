import scipy.io as sio 
import numpy as np
import torch
import os
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from sklearn.metrics import roc_curve, auc

from mri_pet_dataset_test import TestDataset
from gan_models import * 
from densenet import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda = torch.cuda.is_available()

# initial for recurrence
seed = 23
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


SAVE_PATH = './Generated_and_Real_PET_Results'

WORKERS = 0
BATCH_SIZE = 1

dataset = TestDataset()
data_loader_all = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)


G = Generator_Pyconv357_Attention().cuda()
# AD vs CN
G.load_state_dict(torch.load('./Generator_save/Generator_Pyconv357_Attention_with_Task_induced_Discriminator(Main)/'))
# pMCI vs sMCI
# ...

T = Task_induced_Discriminator().cuda()
# AD vs CN
T.load_state_dict(torch.load('./Task_Induced_Discriminator_save/PET/10_TLoss0.1841_TrainACC0.9271_TestACC0.8875_TestSEN0.8958_TestSPE0.8811_TestAUC0.9518_F1S0.8745.pth')) # Can also use others retrained classification models. 
# pMCI vs sMCI
# ...



##################################################################################
#                                   Test
##################################################################################
# G.eval()
# T.eval()
TP = 0
FP = 0
FN = 0
TN = 0
labels = []
scores = []
iteration = 0 
for val_test_data in data_loader_all:
    iteration += 1
    val_test_imgs = val_test_data[0]
    val_test_labels = val_test_data[1]
    val_test_labels_ = Variable(val_test_labels).cuda()
    val_test_data_batch_size = val_test_imgs.size()[0]
    fname = val_test_data[2][0].split('.')[0]

    # Complete dataset with MRI and PET 
    mri_images = val_test_imgs[:, 0, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
    mri_images = Variable(mri_images.cuda(), requires_grad=False)
    pet_images = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
    pet_images = Variable(pet_images.cuda(), requires_grad=False)

    # Incomplete dataset with only MRI 
    # mri_images = val_test_imgs.view(val_test_data_batch_size, 1, 76, 94, 76)
    # mri_images = Variable(mri_images.cuda(), requires_grad=False)

    x_fake = G(mri_images)
    result_c_ = T(x_fake)

    out_c = F.softmax(result_c_, dim=1)
    score = out_c[0][1].data.cpu().item()
    score = round(score, 4)
    scores.append(score)
    _, predicted__ = torch.max(out_c.data, 1)
    PREDICTED = predicted__.data.cpu().numpy()
    REAL = val_test_labels_.data.cpu().numpy()
    labels.append(REAL)

    if PREDICTED == 1 and REAL == 1:
        TP += 1
    elif PREDICTED == 1 and REAL == 0:
        FP += 1
    elif PREDICTED == 0 and REAL == 1:
        FN += 1 
    elif PREDICTED == 0 and REAL == 0:
        TN += 1
    else:
        continue

    fake_data = np.squeeze(x_fake.data.cpu().numpy())
    real_data = np.squeeze(pet_images.data.cpu().numpy())
 
    # Save as ||.nii.gz|| format.
    # file_name1 = os.path.join(SAVE_PATH,'{}_fake.nii.gz'.format(fname))
    # generated_pet = nib.Nifti1Image(fake_data, np.eye(4)) 
    # nib.save(generated_pet, file_name1)
    # file_name2 = os.path.join(SAVE_PATH,'{}_pet.nii.gz'.format(fname))
    # real_pet = nib.Nifti1Image(real_data, np.eye(4)) 
    # nib.save(real_pet, file_name2)


    # Save as ||.mat|| format.
    file_name1 = os.path.join(SAVE_PATH,'{}_fake.mat'.format(fname))
    sio.savemat(file_name1, {'data':fake_data})
    file_name2 = os.path.join(SAVE_PATH,'{}_pet.mat'.format(fname))
    sio.savemat(file_name2, {'data':real_data})


test_acc = (TP + TN)/((TP + TN + FP + FN) +0.00001)
test_sen = TP/((TP + FN)+0.00001)
test_spe = TN/((FP + TN)+0.00001)

fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

print(
        'Test_ACC:{:.4f} {}/{}'.format(round(test_acc, 4), (TP + TN), (TP + TN + FP + FN)),
        'Test_SEN:{:.4f} {}/{}'.format(round(test_sen, 4), TP , (TP + FN)),
        'Test_SPE:{:.4f} {}/{}'.format(round(test_spe, 4), TN, (FP + TN)),
        'Test_AUC:{:.4f}'.format(round(roc_auc, 4) ),
    )