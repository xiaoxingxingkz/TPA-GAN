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

from mri_pet_dataset_train import TrainDataset
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
PET_MODEL_PATH = './Task_Induced_Discriminator_save/PET'
MRI_MODEL_PATH = './Task_Induced_Discriminator_save/MRI'

TRAIN_BATCH_SIZE = 4    
TEST_BATCH_SIZE = 1
LR = 1e-3               
EPOCH = 40
WORKERS = 0

# load train data
dataset_train = TrainDataset()
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size= TRAIN_BATCH_SIZE, shuffle=True, num_workers=WORKERS)
data_loader_valid = torch.utils.data.DataLoader(dataset_train, batch_size= TEST_BATCH_SIZE, shuffle=False, num_workers=WORKERS)

# load test data 
dataset_test = TestDataset()
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size= TEST_BATCH_SIZE, shuffle=False, num_workers=WORKERS)


# loss function
cirterion = nn.CrossEntropyLoss().cuda()
criterion_bce = nn.BCELoss().cuda()

# number of iterations
iter_t = 1

# load model  
T = Task_induced_Discriminator().cuda() # ****** This classification model is densenet21. ******


# load optimazer
t_optimizer = optim.Adam(T.parameters(), lr=LR, weight_decay=1e-4) 
#t_optimizer = optim.SGD(T.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)


################################################################################
#                                  PET
################################################################################
train_pet = 1 #switch 0:stop 1:start
if train_pet == 1:

    for iteration in range(EPOCH):
        start_time = time.time()

        # learning rate decay
        if iteration == 21 or iteration == 31:
            for p in t_optimizer.param_groups:
                p['lr'] = p['lr'] * 0.1
        # for p in t_optimizer.param_groups:
        #     print(p['lr']) 

        ##########################################################
        #                      Train
        ##########################################################
        T.train()
        total_loss = 0
        total_num = 0
        for train_data in data_loader_train:
 
            labels = train_data[1] 
            labels = Variable(labels).cuda()
            images = train_data[0]
            train_batch_size = images.size()[0]
 
            # mri_images = images[:, 0, :, :, :].view(train_batch_size, 1, 76, 94, 76)
            # mri_images = Variable(mri_images.cuda(), requires_grad=False)
            pet_images = images[:, 1, :, :, :].view(train_batch_size, 1, 76, 94, 76)
            pet_images = Variable(pet_images.cuda(), requires_grad=False)

            for itera_f in range(iter_t):
                for p in T.parameters():
                    p.requires_grad = True

                t_optimizer.zero_grad()

                #input
                prediction = T(pet_images)       

                # loss function
                loss_t =  cirterion(prediction, labels) 

                loss_t.backward() 
                t_optimizer.step()

        total_loss += loss_t.item()
        total_num += 1
        ##########################################################
        #                    validation
        ##########################################################
        # T.eval()
        for p in T.parameters():
            p.requires_grad = False

        for train_s in range(1):
            TP_ = 0
            FP_ = 0
            FN_ = 0
            TN_ = 0
            for val_trian_data in data_loader_valid:
                val_trian_imgs = val_trian_data[0]
                val_trian_labels = val_trian_data[1]
                val_trian_labels_ = Variable(val_trian_labels).cuda()
                val_trian_data_batch_size = val_trian_imgs.size()[0]

                # mri_images = val_trian_imgs[:, 0,:, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
                # mri_images = Variable(mri_images.cuda(), requires_grad=False)
                pet_images = val_trian_imgs[:, 1, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
                pet_images = Variable(pet_images.cuda(), requires_grad=False)
                
                #input
                result_c = T(pet_images)
                out_c = F.softmax(result_c, dim=1)
                _, predicted = torch.max(out_c.data, 1)
                PREDICTED_ = predicted.data.cpu().numpy()

                if out_c[:, 1].data.cpu().item() > 0.5:
                    PREDICTED_ = 1
                else:
                    PREDICTED_ = 0
                
                REAL_ = val_trian_labels_.data.cpu().numpy()

                if PREDICTED_ == 1 and REAL_ == 1:
                    TP_ += 1
                elif PREDICTED_ == 1 and REAL_ == 0:
                    FP_ += 1
                elif PREDICTED_ == 0 and REAL_ == 1:
                    FN_ += 1 
                elif PREDICTED_ == 0 and REAL_ == 0:
                    TN_ += 1
                else:
                    continue
        train_acc = (TP_ + TN_)/(TP_ + TN_ + FP_ + FN_)

        ##########################################################
        #                         test
        ##########################################################
        for test_s in range(1):
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

        t_comp = (time.time() - start_time)

        # print log info
        print('[{}/{}]'.format(iteration + 1, EPOCH),
            'Task_classification: {:.4f}'.format(total_loss/total_num),
            'Train_ACC:{:.4f} {}/{}'.format(round(train_acc, 4), (TP_ + TN_), (TP_ + TN_ + FP_ + FN_)),
            'Test_ACC:{:.4f} {}/{}'.format(round(test_acc, 4), (TP + TN), (TP + TN + FP + FN)),
            'Test_SEN:{:.4f} {}/{}'.format(round(test_sen, 4), TP , (TP + FN)),
            'Test_SPE:{:.4f} {}/{}'.format(round(test_spe, 4), TN, (FP + TN)),
            'Test_AUC:{:.4f}'.format(round(roc_auc, 4) ),
            'Test_F1S:{:.4f}'.format(round(test_f1s, 4) ),
            'Time_Taken: {} sec'.format(t_comp)
            )
        #save model
        torch.save(T.state_dict(), os.path.join(PET_MODEL_PATH, '{}_TLoss{}_TrainACC{}_TestACC{}_TestSEN{}_TestSPE{}_TestAUC{}_F1S{}.pth'.format(
            iteration + 1, 
            round(total_loss/total_num, 4), 
            round(train_acc, 4), 
            round(test_acc, 4),
            round(test_sen, 4),
            round(test_spe, 4),
            round(roc_auc, 4),
            round(test_f1s, 4),
            )))


################################################################################
#                  MRI, we make MRI-to-PET generation, so this is not used.
################################################################################
train_mri = 0 #switch 0:stop 1:start
if train_mri == 1:

    for iteration in range(EPOCH):
        start_time = time.time()

        # learning rate decay
        if iteration == 21 or iteration == 31:
            for p in t_optimizer.param_groups:
                p['lr'] = p['lr'] * 0.1
        # for p in t_optimizer.param_groups:
        #     print(p['lr']) 

        ##########################################################
        #                      Train
        ##########################################################
        T.train()
        total_loss = 0
        total_num = 0
        for train_data in data_loader_train:
 
            labels = train_data[1] 
            labels = Variable(labels).cuda()
            images = train_data[0]
            train_batch_size = images.size()[0]
 
            mri_images = images[:, 0, :, :, :].view(train_batch_size, 1, 76, 94, 76)
            mri_images = Variable(mri_images.cuda(), requires_grad=False)
            # pet_images = images[:, 1, :, :, :].view(train_batch_size, 1, 76, 94, 76)
            # pet_images = Variable(pet_images.cuda(), requires_grad=False)

            for itera_f in range(iter_t):
                for p in T.parameters():
                    p.requires_grad = True

                t_optimizer.zero_grad()

                #input
                prediction = T(mri_images)       

                # loss function
                loss_t =  cirterion(prediction, labels) 

                loss_t.backward() 
                t_optimizer.step()

        total_loss += loss_t.item()
        total_num += 1
        ##########################################################
        #                    validation
        ##########################################################
        # T.eval()
        for p in T.parameters():
            p.requires_grad = False

        for train_s in range(1):
            TP_ = 0
            FP_ = 0
            FN_ = 0
            TN_ = 0
            for val_trian_data in data_loader_valid:
                val_trian_imgs = val_trian_data[0]
                val_trian_labels = val_trian_data[1]
                val_trian_labels_ = Variable(val_trian_labels).cuda()
                val_trian_data_batch_size = val_trian_imgs.size()[0]

                mri_images = val_trian_imgs[:, 0,:, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
                mri_images = Variable(mri_images.cuda(), requires_grad=False)
                # pet_images = val_trian_imgs[:, 1, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
                # pet_images = Variable(pet_images.cuda(), requires_grad=False)
                
                #input
                result_c = T(mri_images)
                out_c = F.softmax(result_c, dim=1)
                _, predicted = torch.max(out_c.data, 1)
                PREDICTED_ = predicted.data.cpu().numpy()

                if out_c[:, 1].data.cpu().item() > 0.5:
                    PREDICTED_ = 1
                else:
                    PREDICTED_ = 0
                
                REAL_ = val_trian_labels_.data.cpu().numpy()

                if PREDICTED_ == 1 and REAL_ == 1:
                    TP_ += 1
                elif PREDICTED_ == 1 and REAL_ == 0:
                    FP_ += 1
                elif PREDICTED_ == 0 and REAL_ == 1:
                    FN_ += 1 
                elif PREDICTED_ == 0 and REAL_ == 0:
                    TN_ += 1
                else:
                    continue
        train_acc = (TP_ + TN_)/(TP_ + TN_ + FP_ + FN_)

        ##########################################################
        #                         test
        ##########################################################
        for test_s in range(1):
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

                mri_images_ = val_test_imgs[:, 0, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
                mri_images_ = Variable(mri_images_.cuda(), requires_grad=False)
                # pet_images_ = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
                # pet_images_ = Variable(pet_images_.cuda(), requires_grad=False)

                # input
                result_c_ = T(mri_images_)
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

        t_comp = (time.time() - start_time)

        # print log info
        print('[{}/{}]'.format(iteration + 1, EPOCH),
            'Task_classification: {:.4f}'.format(total_loss/total_num),
            'Train_ACC:{:.4f} {}/{}'.format(round(train_acc, 4), (TP_ + TN_), (TP_ + TN_ + FP_ + FN_)),
            'Test_ACC:{:.4f} {}/{}'.format(round(test_acc, 4), (TP + TN), (TP + TN + FP + FN)),
            'Test_SEN:{:.4f} {}/{}'.format(round(test_sen, 4), TP , (TP + FN)),
            'Test_SPE:{:.4f} {}/{}'.format(round(test_spe, 4), TN, (FP + TN)),
            'Test_AUC:{:.4f}'.format(round(roc_auc, 4) ),
            'Test_F1S:{:.4f}'.format(round(test_f1s, 4) ),
            'Time_Taken: {} sec'.format(t_comp)
            )
        #save model
        torch.save(T.state_dict(), os.path.join(MRI_MODEL_PATH, '{}_TLoss{}_TrainACC{}_TestACC{}_TestSEN{}_TestSPE{}_TestAUC{}_F1S{}.pth'.format(
            iteration + 1, 
            round(total_loss/total_num, 4), 
            round(train_acc, 4), 
            round(test_acc, 4),
            round(test_sen, 4),
            round(test_spe, 4),
            round(roc_auc, 4),
            round(test_f1s, 4),
            )))
