import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset

class TestDataset(Dataset):
    def __init__(self, image_path = './ADNI2_MRI_PET_ADNC_255'):   # Pixel range 0~255 
        self.path = image_path
        test_data = []
        label = os.listdir(self.path)
        for image_label in label:
            test_data.append(image_label)
        test_data = np.asarray(test_data)
        self.name = test_data
    
    def __len__(self):
        return len(self.name)
        
    def __getitem__(self, index):
        file_name = self.name[index]                
        path = os.path.join(self.path, file_name)
        label = int(file_name[0]) 
        img = sio.loadmat(path)
        out = img['data']
        data = torch.from_numpy(out).float()        
        num = len(self.name)

        return data, label, file_name
