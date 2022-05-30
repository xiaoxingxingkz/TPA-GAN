import numpy as np
import torch
import os
import cv2
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.autograd import Variable
import time
from ssim_metric import SSIM 
from sklearn.metrics import roc_curve, auc
import math


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda = torch.cuda.is_available()


from mri_pet_dataset_train import TrainDataset
from mri_pet_dataset_test import TestDataset
from gan_models import *
from densenet import *

# initial for recurrence
seed = 23
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


# initial setup
MODEL_PATH = './Generator_save/Generator_Pyconv357_Attention_with_Task_induced_Discriminator(Main)'
ISHOWN_PATH = './Train_results'
ISHOWN_PATH_ = './Test_results'
# LOSS_PATH = './Loss'
TRAIN_BATCH_SIZE = 2 
TEST_BATCH_SIZE = 1
LR = 1e-4
EPOCH = 300
WORKERS = 0

# load train data
dataset_train = TrainDataset()
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = TRAIN_BATCH_SIZE, shuffle = True, num_workers = WORKERS)
data_loader_valid = torch.utils.data.DataLoader(dataset_train, batch_size = TEST_BATCH_SIZE, shuffle = False, num_workers = WORKERS)


# load test data 
dataset_test = TestDataset()
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = TEST_BATCH_SIZE, shuffle = False, num_workers = WORKERS)


criterion_bce = nn.BCELoss().cuda()
criterion_l1 = nn.L1Loss().cuda()
criterion_mse =nn.MSELoss().cuda()
cirterion_ssim = SSIM().cuda()
cirterion = nn.CrossEntropyLoss().cuda()


# number of iterations
iter_g = 1
iter_d = 1
iter_t = 0

G = Generator_Pyconv357_Attention().cuda()
D = Standard_Discriminator().cuda()
T = Task_induced_Discriminator().cuda()


T.load_state_dict(torch.load('./Task_Induced_Discriminator_save/PET/10_TLoss0.1841_TrainACC0.9271_TestACC0.8875_TestSEN0.8958_TestSPE0.8811_TestAUC0.9518_F1S0.8745.pth'))
# T.eval() # Annotate this, use parameters of training set.

g_optimizer = optim.Adam(G.parameters(), lr=0.0001) 
d_optimizer = optim.Adam(D.parameters(), lr=0.0004)
t_optimizer = optim.Adam(T.parameters(), lr=LR, weight_decay = 1e-4)


'''
########################################################################################################
########################################################################################################
####################################### Processing of Generation #######################################
######################################################################################################## 
########################################################################################################
'''
# loss_filename = os.path.join(LOSS_PATH, 'loss_data.txt')
# fw = open(loss_filename, 'w')  

for iteration in range(EPOCH):

    print(iteration + 1)
    start_time = time.time()
    G.train()

    # if iteration >= 200:
    #     for p in t_optimizer.param_groups:
    #         p['lr'] *= 0.9
    

    ##########################################################
    #                         Train
    ##########################################################
    total_loss = 0
    total_num = 0
    for train_data in data_loader_train:

        labels = train_data[1]
        labels_ = Variable(labels).cuda()
        images = train_data[0]
        train_batch_size = images.size()[0]

        mri_images = images[:, 0, :, :, :].view(train_batch_size, 1, 76, 94, 76)
        mri_images = Variable(mri_images.cuda(), requires_grad=False)
        pet_images = images[:, 1, :, :, :].view(train_batch_size, 1, 76, 94, 76)
        pet_images = Variable(pet_images.cuda(), requires_grad=False)
    
        real_y = Variable(torch.ones((train_batch_size, 1)).cuda())
        fake_y = Variable(torch.zeros((train_batch_size, 1)).cuda())
        ##########################################################
        #                      Generator
        ##########################################################
        for itera_g in range(iter_g):
            for p in G.parameters():
                p.requires_grad = True
            for p in D.parameters():
                p.requires_grad = False
            g_optimizer.zero_grad()

            x_fake = G(mri_images)
            d_ = D(x_fake)
            t_ = T(x_fake) 

            # loss function
            loss1 = criterion_l1(x_fake, pet_images) 
            loss2 = criterion_mse(x_fake, pet_images) 
            loss3 = cirterion_ssim(x_fake, pet_images)  
            a = loss1.cpu().data
            b = loss2.cpu().data
            c = loss3.cpu().data
            
            max_value = max(a, b, c)
            a_value = int(math.log(max_value/a, 10))
            b_value = int(math.log(max_value/b, 10))
            c_value = int(math.log(max_value/c, 10))
        
            theta_a = 1
            theta_b = 1
            theta_c = 1

            if a_value > 0:
                theta_a = 10**a_value
            if b_value > 0:
                theta_b = 10**b_value
            if c_value > 0:
                theta_c = 10**c_value

            generator_loss = theta_a * loss1 + theta_b * loss2 + theta_c * loss3 
            standard_discriminator_loss = criterion_bce(d_, real_y[:train_batch_size])
            task_induced_discriminator_loss = cirterion(t_, labels_)

            if iteration < 40:
                loss_g = generator_loss
            elif iteration >= 40 and iteration <= 80:
                loss_g = generator_loss + standard_discriminator_loss 
            else:
                loss_g = generator_loss + task_induced_discriminator_loss
            
            loss_g.backward()
            g_optimizer.step()
        total_loss += loss_g.item()
        total_num += 1
        ##########################################################
        #                   Standard-Discriminator
        ##########################################################
        if iteration >= 40 and iteration <= 80:
            for itera_d in range(iter_d):
                for p in G.parameters():
                    p.requires_grad = False
                for p in D.parameters():
                    p.requires_grad = True
                d_optimizer.zero_grad()

                x_fake = G(mri_images)
                #z_rand = Variable(torch.randn(train_batch_size, 1, 76, 94, 76).cuda(), requires_grad=False)
                #z_noise = G(z_rand)

                dx = D(x_fake)
                dy = D(pet_images)
                #dz = D(z_noise)

                # loss function 
                x_fake_loss = criterion_bce(dx, fake_y[:train_batch_size])
                y_real_loss = criterion_bce(dy, real_y[:train_batch_size])                
                #z_random_loss = criterion_bce(dz, fake_y[:train_batch_size])

                loss_d = x_fake_loss + y_real_loss 
                loss_d.backward() 
                d_optimizer.step()

  

    ##########################################################
    #                       Validation
    ##########################################################
    # G.eval()
    for p in G.parameters():
        p.requires_grad = False
    for p in D.parameters():
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

            mri_images = val_trian_imgs[:, 0, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
            mri_images = Variable(mri_images.cuda(), requires_grad=False)
            pet_images = val_trian_imgs[:, 1, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
            pet_images = Variable(pet_images.cuda(), requires_grad=False)

            gen_pet_images = G(mri_images)
            result_c = T(gen_pet_images)

            out_c = F.softmax(result_c, dim=1)
            _, predicted = torch.max(out_c.data, 1)
            PREDICTED_ = predicted.data.cpu().numpy()
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
        ori_data = np.squeeze(mri_images.data.cpu().numpy())
        fake_data = np.squeeze(gen_pet_images.data.cpu().numpy())
        real_data = np.squeeze(pet_images.data.cpu().numpy())
        if len(fake_data.shape) == 3:
            for i in range(76):
                img_2d = fake_data[:, :, i]
                #img_2d = img_2d * 255
                img_fileName =  str(i) + '_generate.png'
                end_path = os.path.join(ISHOWN_PATH, img_fileName)
                cv2.imwrite(end_path, img_2d)

                img_2d_r = real_data[:, :, i]
                #img_2d_r = img_2d_r * 255
                img_fileName_r =  str(i) + '_pet.png'
                end_path_r = os.path.join(ISHOWN_PATH, img_fileName_r)
                cv2.imwrite(end_path_r, img_2d_r)

                img_2d_o = ori_data[:, :, i]
                #img_2d_o = img_2d_o * 255
                img_fileName_o =  str(i) + '_mri.png'
                end_path_o = os.path.join(ISHOWN_PATH, img_fileName_o)
                cv2.imwrite(end_path_o, img_2d_o)                

        elif len(fake_data.shape) == 4:
            for i in range(76):
                img_2d = fake_data[0, :, :, i]
                img_fileName =  str(i) + '_generate.png'
                end_path = os.path.join(ISHOWN_PATH, img_fileName)
                cv2.imwrite(end_path, img_2d)

                img_2d_r = real_data[0, :, :, i]
                img_fileName_r =  str(i) + '_pet.png'
                end_path_r = os.path.join(ISHOWN_PATH, img_fileName_r)
                cv2.imwrite(end_path_r, img_2d_r)

                img_2d_o = ori_data[:, :, i]
                img_fileName_o =  str(i) + '_mri.png'
                end_path_o = os.path.join(ISHOWN_PATH, img_fileName_o)
                cv2.imwrite(end_path_o, img_2d_o)


    ##########################################################
    #                        Test
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
            pet_images_ = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
            pet_images_ = Variable(pet_images_.cuda(), requires_grad=False)


            gen_pet_images = G(mri_images_)
            result_c_ = T(gen_pet_images)

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

        test_acc = (TP + TN)/(TP + TN + FP + FN)
        test_sen = TP/(TP + FN)
        test_spe = TN/(FP + TN)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        ori_data = np.squeeze(mri_images_.data.cpu().numpy())
        fake_data = np.squeeze(gen_pet_images.data.cpu().numpy())
        real_data = np.squeeze(pet_images_.data.cpu().numpy())
        if len(fake_data.shape) == 3:
            for i in range(76):
                img_2d = fake_data[:, :, i]
                img_fileName =  str(i) + '_generate.png'
                end_path = os.path.join(ISHOWN_PATH_, img_fileName)
                cv2.imwrite(end_path, img_2d)

                img_2d_r = real_data[:, :, i]
                img_fileName_r =  str(i) + '_pet.png'
                end_path_r = os.path.join(ISHOWN_PATH_, img_fileName_r)
                cv2.imwrite(end_path_r, img_2d_r)

                img_2d_o = ori_data[:, :, i]
                img_fileName_o =  str(i) + '_mri.png'
                end_path_o = os.path.join(ISHOWN_PATH_, img_fileName_o)
                cv2.imwrite(end_path_o, img_2d_o)

        elif len(fake_data.shape) == 4:
            for i in range(76):
                img_2d = fake_data[0, :, :, i]
                img_fileName =  str(i) + '_generate.png'
                end_path = os.path.join(ISHOWN_PATH_, img_fileName)
                cv2.imwrite(end_path, img_2d)

                img_2d_r = real_data[0, :, :, i]
                img_fileName_r =  str(i) + '_pet.png'
                end_path_r = os.path.join(ISHOWN_PATH_, img_fileName_r)
                cv2.imwrite(end_path_r, img_2d_r)

                img_2d_o = ori_data[0, :, :, i]
                img_fileName_o =  str(i) + '_mri.png'
                end_path_o = os.path.join(ISHOWN_PATH_, img_fileName_o)
                cv2.imwrite(end_path_o, img_2d_o)                
    t_comp = (time.time() - start_time)


    #print log info
    print('[{}/{}]'.format(iteration + 1, EPOCH),
         'Generator_Loss: {:.4f}'.format(total_loss/total_num),
         #'Standard_Discriminator_Loss: {:.4f}'.format(loss_d.item()),
         'Train_ACC:{:.4f} {}/{}'.format(round(train_acc, 4), (TP_ + TN_), (TP_ + TN_ + FP_ + FN_)),
         'Test_ACC:{:.4f} {}/{}'.format(round(test_acc, 4), (TP + TN), (TP + TN + FP + FN)),
         'Test_SEN:{:.4f} {}/{}'.format(round(test_sen, 4), TP , (TP + FN)),
         'Test_SPE:{:.4f} {}/{}'.format(round(test_spe, 4), TN, (FP + TN)),
         'Test_AUC:{:.4f}'.format(round(roc_auc, 4) ),
         'Time_Taken: {} sec'.format(t_comp)
         )    
    #fw.write(str(loss_g) + '\n')
    
    #save model
    torch.save(G.state_dict(), os.path.join(MODEL_PATH, '{}_GLoss{}_TrainACC{}_TestACC{}_TestSEN{}_TestSPE{}_TestAUC{}_G.pth'.format(
                iteration + 1,
                round(total_loss/total_num, 4), 
                round(train_acc, 4), 
                round(test_acc, 4),
                round(test_sen, 4),
                round(test_spe, 4),
                round(roc_auc, 4),
              )))
    # torch.save(D.state_dict(), os.path.join(MODEL_PATH, '{}_GLoss{}_TrainACC{}_TestACC{}_TestSEN{}_TestSPE{}_TestAUC{}_D.pth'.format(
    #             iteration + 1, 
    #             round(total_loss/total_num, 4), 
    #             round(train_acc, 4), 
    #             round(test_acc, 4),
    #             round(test_sen, 4),
    #             round(test_spe, 4),
    #             round(roc_auc, 4),
    #           )))

# fw.close()
'''
########################################################################################################
########################################################################################################
###################################### Processing of Generating ended ##################################
######################################################################################################## 
########################################################################################################
'''
