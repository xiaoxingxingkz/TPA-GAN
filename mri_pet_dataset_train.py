import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset


class TrainDataset(Dataset):
    def __init__(self, image_path = './ADNI1_MRI_PET_ADNC_255'): # Pixel range 0~255 
        self.path = image_path
        train_data = []
        label = os.listdir(self.path)
        for image_label in label:
            train_data.append(image_label)
        train_data = np.asarray(train_data)
        self.name = train_data

    def __len__(self):
        return len(self.name)
        
    def __getitem__(self, index):
        file_name = self.name[index]                    
        path = os.path.join(self.path, file_name)
        label = int(file_name[0]) 
        img = sio.loadmat(path)
        out = img['data']
        data = torch.from_numpy(out).float()          

        return data, label
