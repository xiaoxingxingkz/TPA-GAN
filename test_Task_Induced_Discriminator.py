import numpy as np
import random
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.autograd import Variable
import time
from sklearn.metrics import roc_curve, auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = torch.cuda.is_available()

from mri_pet_dataset_test import TestDataset
from densenet import *


# initial for recurrence
seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


# initial setup
TEST_BATCH_SIZE = 1
WORKERS = 0
# load test data 
dataset_test = TestDataset()
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size= TEST_BATCH_SIZE, shuffle=False, num_workers=WORKERS)


# load Task induced Discriminator model of AD vs CN for PET images
T = Task_induced_Discriminator().cuda() 
T.load_state_dict(torch.load('./Task_Induced_Discriminator_save/PET/10_TLoss0.1841_TrainACC0.9271_TestACC0.8875_TestSEN0.8958_TestSPE0.8811_TestAUC0.9518_F1S0.8745.pth'))




################################################################################
#                                  PET TEST
################################################################################
# T.eval() # Annotate this, use parameters of training set.
TP = 0
FP = 0
FN = 0
TN = 0
labels = []
scores = []
for val_test_data in data_loader_test:

    val_test_imgs = val_test_data[0]
    val_test_labels = val_test_data[1]
    val_test_labels_ = Variable(val_test_labels).cuda()
    val_test_data_batch_size = val_test_imgs.size()[0]

    # mri_images_ = val_test_imgs[:, 0, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
    # mri_images_ = Variable(mri_images_.cuda(), requires_grad=False)
    pet_images_ = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
    pet_images_ = Variable(pet_images_.cuda(), requires_grad=False)

    # input
    result_c_ = T(pet_images_)
    out_c = F.softmax(result_c_, dim=1)
    _, predicted__ = torch.max(out_c.data, 1)
    PREDICTED = predicted__.data.cpu().numpy()
    score = out_c[:,1].data.cpu().item()
    score = round(score, 4)
    scores.append(score)

    # if score > 0.5:
    #     PREDICTED = 1
    # else:
    #     PREDICTED = 0

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

test_acc = (TP + TN)/(TP + TN + FP + FN)
test_sen = TP/(TP + FN)
test_spe = TN/(FP + TN)

fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

test_f1s = 2*(TP/(TP + FP + 0.0001))*(TP/(TP + FN + 0.0001))/((TP/(TP + FP + 0.0001)) + (TP/(TP + FN + 0.0001)) + 0.0001)

# print log info
print(
    'Test_ACC:{:.4f} {}/{}'.format(round(test_acc, 4), (TP + TN), (TP + TN + FP + FN)),
    'Test_SEN:{:.4f} {}/{}'.format(round(test_sen, 4), TP , (TP + FN)),
    'Test_SPE:{:.4f} {}/{}'.format(round(test_spe, 4), TN, (FP + TN)),
    'Test_AUC:{:.4f}'.format(round(roc_auc, 4) ),
    'Test_F1S:{:.4f}'.format(round(test_f1s, 4) ),
)

