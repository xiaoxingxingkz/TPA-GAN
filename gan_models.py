import torch
import torch.nn as nn
from torch.nn import functional as F

class Generator_Basic(nn.Module):
    """Generator Unet structure simplified version"""

    def __init__(self, conv_dim=64):
        super(Generator_Basic, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()

        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=4, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv3 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv4 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=4, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*8)        

        self.tp_conv5 = nn.ConvTranspose3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=2, padding=0, output_padding=0, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.ConvTranspose3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=2, padding=0, output_padding=0, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)       
        self.tp_conv7 = nn.ConvTranspose3d(conv_dim*6, conv_dim*3, kernel_size=3, stride=2, padding=(1, 0, 1), output_padding=(1, 0, 1), bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*3)
        self.tp_conv8 = nn.ConvTranspose3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv9 = nn.Conv3d(conv_dim*2, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim)
        self.tp_conv10 = nn.Conv3d(conv_dim, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, z):
   

        h = self.tp_conv1(z)
        h = self.relu(self.bn1(h))
        skI = h

        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))
        skII = h
        
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))
        skIII = h 

        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = self.tp_conv5(h)
        h = self.relu(self.bn5(h))
        skip3 = torch.cat([h, skIII], 1)

        h = self.tp_conv6(skip3)
        h = self.relu(self.bn6(h))
        skip2 = torch.cat([h, skII], 1)
   
        h = self.tp_conv7(skip2)
        h = self.relu(self.bn7(h))
        skip1 =  torch.cat([h, skI], 1)  

        h = self.tp_conv8(skip1)
        h = self.relu(self.bn8(h))

        h = self.tp_conv9(h)
        h = self.relu(self.bn9(h))
        h = self.tp_conv10(h)

        return h

class Generator_Pyconv357(nn.Module):
    """Generator Unet structure"""
    def __init__(self, conv_dim=8):
        super(Generator_Pyconv357, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.down_sampling = nn.MaxPool3d(kernel_size=3, stride=2)

        #Encoder
        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim)

        self.tp_conv3 = nn.Conv3d(conv_dim*3, conv_dim*6, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv4 = nn.Conv3d(conv_dim*6, conv_dim*6, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2)  

        self.tp_conv5 = nn.Conv3d(conv_dim*12, conv_dim*24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.Conv3d(conv_dim*24, conv_dim*24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)  

        self.tp_conv7 = nn.Conv3d(conv_dim*24, conv_dim*48, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv8 = nn.Conv3d(conv_dim*48, conv_dim*48, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*8)  

        self.rbn = nn.Conv3d(conv_dim*48, conv_dim*48, kernel_size=3, stride=1, padding=1, bias=True)

        #Decoder
        self.tp_conv9 = nn.ConvTranspose3d(conv_dim*48, conv_dim*24, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv10 = nn.Conv3d(conv_dim*48, conv_dim*24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn10 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv11 = nn.Conv3d(conv_dim*24, conv_dim*24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn11 = nn.BatchNorm3d(conv_dim*4)

        self.tp_conv12 = nn.ConvTranspose3d(conv_dim*24, conv_dim*12, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(0, 1, 0), bias=True)
        self.bn12 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv13 = nn.Conv3d(conv_dim*24, conv_dim*12, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn13 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv14 = nn.Conv3d(conv_dim*12, conv_dim*12, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn14 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv15 = nn.ConvTranspose3d(conv_dim*12, conv_dim*3, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(1, 1, 1), bias=True)
        self.bn15 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv16 = nn.Conv3d(conv_dim*6, conv_dim*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn16 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv17 = nn.Conv3d(conv_dim*3, conv_dim*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn17 = nn.BatchNorm3d(conv_dim*1)

        self.conv_an_0 = nn.Conv3d(conv_dim*3, conv_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_an_1 = nn.Conv3d(conv_dim*3, conv_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        self.convback = nn.Conv3d(conv_dim*3, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)

        self.tp_conv18 = nn.Conv3d(conv_dim*3, 1, kernel_size=3, stride=1, padding=1, bias=True)

        #add pyconv module
        self.conv1_7 = nn.Conv3d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv2_7 = nn.Conv3d(conv_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv1_5 = nn.Conv3d(1, conv_dim, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2_5 = nn.Conv3d(conv_dim, conv_dim, kernel_size=5, stride=1, padding=2, bias=True)


        self.conv3_5 = nn.Conv3d(conv_dim*3, conv_dim*6, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4_5 = nn.Conv3d(conv_dim*6, conv_dim*6, kernel_size=5, stride=1, padding=2, bias=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
   

        h = self.tp_conv1(z)
        h = self.tp_conv2(self.relu(h))
        j = self.conv1_7(z)
        j = self.conv2_7(self.relu(j))
        k = self.conv1_5(z)
        k = self.conv2_5(self.relu(k))
        hj = torch.cat([h, j], 1)
        skip3 = torch.cat([hj, k], 1)  #conv_dim*3
        h = self.down_sampling(self.relu(skip3))


        h2 = self.tp_conv3(h)
        h2 = self.tp_conv4(self.relu(h2))
        q = self.conv3_5(h)
        q = self.conv4_5(self.relu(q))
        skip2 = torch.cat([h2, q], 1) #conv_dim*12
        h = self.down_sampling(self.relu(skip2))

        h = self.tp_conv5(h)
        h = self.tp_conv6(self.relu(h))
        skip1 = h                     #conv_dim*24
        h = self.down_sampling(self.relu(h))        

        h = self.tp_conv7(h)
        h = self.tp_conv8(self.relu(h))
        c1 = h       

        #RNB
        h = self.rbn(self.relu(c1))
        h = self.rbn(self.relu(h))
        c2 = h + c1

        h = self.rbn(self.relu(c2))
        h = self.rbn(self.relu(h))
        c3 = h + c2

        h = self.rbn(self.relu(c3))
        h = self.rbn(self.relu(h))
        c4 = h + c3        

        h = self.rbn(self.relu(c4))
        h = self.rbn(self.relu(h))
        c5 = h + c4

        h = self.rbn(self.relu(c5))
        h = self.rbn(self.relu(h))
        c6 = h + c5

        h = self.rbn(self.relu(c6))
        h = self.rbn(self.relu(h))
        c7 = h + c6
        #RBN

        h = self.tp_conv9(self.relu(c7))
        h = torch.cat([h, skip1], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv10(h))
        h = self.relu(self.tp_conv11(h))

        h = self.tp_conv12(h)
        h = torch.cat([h, skip2], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv13(h))
        h = self.relu(self.tp_conv14(h))

        h = self.tp_conv15(h)
        h = torch.cat([h, skip3], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv16(h))
        h = self.relu(self.tp_conv17(h))

        h = self.tp_conv18(h)

        return h

class Generator_Attention(nn.Module):
    """Generator Unet structure"""

    def __init__(self, conv_dim=8):
        super(Generator_Attention, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.down_sampling = nn.MaxPool3d(kernel_size=3, stride=2)

        #Encoder
        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim)

        self.tp_conv3 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv4 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2)  

        self.tp_conv5 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)  

        self.tp_conv7 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv8 = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*8)  

        self.rbn = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)

        #Decoder
        self.tp_conv9 = nn.ConvTranspose3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv10 = nn.Conv3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn10 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv11 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn11 = nn.BatchNorm3d(conv_dim*4)

        self.tp_conv12 = nn.ConvTranspose3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(0, 1, 0), bias=True)
        self.bn12 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv13 = nn.Conv3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn13 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv14 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn14 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv15 = nn.ConvTranspose3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(1, 1, 1), bias=True)
        self.bn15 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv16 = nn.Conv3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn16 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv17 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn17 = nn.BatchNorm3d(conv_dim*1)

        self.conv_an_0 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_an_1 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=1, stride=1, padding=0, bias=True)
        self.convback = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)

        self.tp_conv18 = nn.Conv3d(conv_dim*1, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
   

        h = self.tp_conv1(z)
        h = self.tp_conv2(self.relu(h))
        skip3 = h
        h = self.down_sampling(self.relu(h))

        h = self.tp_conv3(h)
        h = self.tp_conv4(self.relu(h))
        skip2 = h
        h = self.down_sampling(self.relu(h))

        h = self.tp_conv5(h)
        h = self.tp_conv6(self.relu(h))
        skip1 = h
        h = self.down_sampling(self.relu(h))        

        h = self.tp_conv7(h)
        h = self.tp_conv8(self.relu(h))
        c1 = h       

        #RNB
        h = self.rbn(self.relu(c1))
        h = self.rbn(self.relu(h))
        c2 = h + c1

        h = self.rbn(self.relu(c2))
        h = self.rbn(self.relu(h))
        c3 = h + c2

        h = self.rbn(self.relu(c3))
        h = self.rbn(self.relu(h))
        c4 = h + c3        

        h = self.rbn(self.relu(c4))
        h = self.rbn(self.relu(h))
        c5 = h + c4

        h = self.rbn(self.relu(c5))
        h = self.rbn(self.relu(h))
        c6 = h + c5

        h = self.rbn(self.relu(c6))
        h = self.rbn(self.relu(h))
        c7 = h + c6
        #RBN

        h = self.tp_conv9(self.relu(c7))
        h = torch.cat([h, skip1], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv10(h))
        h = self.relu(self.tp_conv11(h))

        h = self.tp_conv12(h)
        h = torch.cat([h, skip2], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv13(h))
        h = self.relu(self.tp_conv14(h))

        h = self.tp_conv15(h)
        h = torch.cat([h, skip3], 1)
        h = self.relu(h)
        h_end = self.relu(self.tp_conv16(h))
        h = self.relu(self.conv_an_0(h_end))
        h = self.conv_an_1(h)
        h_sigmoid = self.sigmoid(h)

        h = self.tp_conv17(h_end)
        h = h * h_sigmoid 

        self.relu(h)
        self.convback(h)

        h = self.tp_conv18(h)

        return h

class Generator_Pyconv357_Attention(nn.Module):
    """Generator Unet structure"""

    def __init__(self, conv_dim=8):
        super(Generator_Pyconv357_Attention, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.down_sampling = nn.MaxPool3d(kernel_size=3, stride=2)

        #Encoder
        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim)

        self.tp_conv3 = nn.Conv3d(conv_dim*3, conv_dim*6, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv4 = nn.Conv3d(conv_dim*6, conv_dim*6, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2)  

        self.tp_conv5 = nn.Conv3d(conv_dim*12, conv_dim*24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.Conv3d(conv_dim*24, conv_dim*24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)  

        self.tp_conv7 = nn.Conv3d(conv_dim*24, conv_dim*48, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv8 = nn.Conv3d(conv_dim*48, conv_dim*48, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*8)  

        self.rbn = nn.Conv3d(conv_dim*48, conv_dim*48, kernel_size=3, stride=1, padding=1, bias=True)

        #Decoder
        self.tp_conv9 = nn.ConvTranspose3d(conv_dim*48, conv_dim*24, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv10 = nn.Conv3d(conv_dim*48, conv_dim*24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn10 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv11 = nn.Conv3d(conv_dim*24, conv_dim*24, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn11 = nn.BatchNorm3d(conv_dim*4)

        self.tp_conv12 = nn.ConvTranspose3d(conv_dim*24, conv_dim*12, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(0, 1, 0), bias=True)
        self.bn12 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv13 = nn.Conv3d(conv_dim*24, conv_dim*12, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn13 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv14 = nn.Conv3d(conv_dim*12, conv_dim*12, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn14 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv15 = nn.ConvTranspose3d(conv_dim*12, conv_dim*3, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(1, 1, 1), bias=True)
        self.bn15 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv16 = nn.Conv3d(conv_dim*6, conv_dim*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn16 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv17 = nn.Conv3d(conv_dim*3, conv_dim*3, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn17 = nn.BatchNorm3d(conv_dim*1)

        self.conv_an_0 = nn.Conv3d(conv_dim*3, conv_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_an_1 = nn.Conv3d(conv_dim*3, conv_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        self.convback = nn.Conv3d(conv_dim*3, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)

        self.tp_conv18 = nn.Conv3d(conv_dim*3, 1, kernel_size=3, stride=1, padding=1, bias=True)

        #add pyconv module
        self.conv1_7 = nn.Conv3d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv2_7 = nn.Conv3d(conv_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv1_5 = nn.Conv3d(1, conv_dim, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2_5 = nn.Conv3d(conv_dim, conv_dim, kernel_size=5, stride=1, padding=2, bias=True)


        self.conv3_5 = nn.Conv3d(conv_dim*3, conv_dim*6, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4_5 = nn.Conv3d(conv_dim*6, conv_dim*6, kernel_size=5, stride=1, padding=2, bias=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, z):
   

        h = self.tp_conv1(z)
        h = self.tp_conv2(self.relu(h))
        j = self.conv1_7(z)
        j = self.conv2_7(self.relu(j))
        k = self.conv1_5(z)
        k = self.conv2_5(self.relu(k))
        hj = torch.cat([h, j], 1)
        skip3 = torch.cat([hj, k], 1)  #conv_dim*3
        h = self.down_sampling(self.relu(skip3))


        h2 = self.tp_conv3(h)
        h2 = self.tp_conv4(self.relu(h2))
        q = self.conv3_5(h)
        q = self.conv4_5(self.relu(q))
        skip2 = torch.cat([h2, q], 1) #conv_dim*12
        h = self.down_sampling(self.relu(skip2))

        h = self.tp_conv5(h)
        h = self.tp_conv6(self.relu(h))
        skip1 = h                     #conv_dim*24
        h = self.down_sampling(self.relu(h))        

        h = self.tp_conv7(h)
        h = self.tp_conv8(self.relu(h))
        c1 = h       

        #RNB
        h = self.rbn(self.relu(c1))
        h = self.rbn(self.relu(h))
        c2 = h + c1

        h = self.rbn(self.relu(c2))
        h = self.rbn(self.relu(h))
        c3 = h + c2

        h = self.rbn(self.relu(c3))
        h = self.rbn(self.relu(h))
        c4 = h + c3        

        h = self.rbn(self.relu(c4))
        h = self.rbn(self.relu(h))
        c5 = h + c4

        h = self.rbn(self.relu(c5))
        h = self.rbn(self.relu(h))
        c6 = h + c5

        h = self.rbn(self.relu(c6))
        h = self.rbn(self.relu(h))
        c7 = h + c6
        #RBN

        h = self.tp_conv9(self.relu(c7))
        h = torch.cat([h, skip1], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv10(h))
        h = self.relu(self.tp_conv11(h))

        h = self.tp_conv12(h)
        h = torch.cat([h, skip2], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv13(h))
        h = self.relu(self.tp_conv14(h))

        h = self.tp_conv15(h)
        h = torch.cat([h, skip3], 1)
        h = self.relu(h)
        h_end = self.relu(self.tp_conv16(h))
        h = self.relu(self.conv_an_0(h_end))
        h = self.conv_an_1(h)
        h_sigmoid = self.sigmoid(h)

        h = self.tp_conv17(h_end)
        h = h * h_sigmoid 

        h = self.tp_conv18(h)

        return h

class Standard_Discriminator(nn.Module):
    def __init__(self, conv_dim=32, out_class=1, is_dis=True):
        super(Standard_Discriminator, self).__init__()
        self.is_dis = is_dis
        self.conv_dim =conv_dim
        n_class = out_class

        self.conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 =nn.BatchNorm3d(conv_dim)
        self.conv2 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim*2)
        self.conv3 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*4)
        self.conv4 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*8) 
        self.conv5 = nn.Conv3d(conv_dim*8, conv_dim*16, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*16) 
        self.conv6 = nn.Conv3d(conv_dim*16, n_class, kernel_size=3, stride=1, padding=0, bias=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn5(self.conv5(h)), negative_slope=0.2)
        h = self.conv6(h)
        if self.is_dis:
            output = nn.Sigmoid()(h.view(h.size()[0], -1))
        else:
            output = h
        return output

