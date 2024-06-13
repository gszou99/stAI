import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .utils import location_to_edge, cross_dist

class spatialDataset(Dataset):
    
    def __init__(self, data_fit, data_supervision, location):
        assert (data_supervision is None or len(data_fit) == len(data_supervision)) and len(data_fit) == len(location)
        self.data_fit = data_fit
        self.data_supervision = data_supervision
        self.location = location
    
    def __len__(self):
        return len(self.data_fit)
    
    def __getitem__(self, index):
        if self.data_supervision is None:
            return self.data_fit[index], None, self.location[index]
        else:
            return self.data_fit[index], self.data_supervision[index], self.location[index]

class rnaDataset(Dataset):
    
    def __init__(self, data_fit, data_supervision, label):
        self.data_fit = data_fit
        self.data_supervision = data_supervision
        self.label = label
    
    def __len__(self):
        return len(self.data_fit)
    
    def __getitem__(self, index):
        return self.data_fit[index], self.data_supervision[index], self.label[index]

class spatialCollate():
    
    def __init__(self, knn = 10, device='cpu'):
        self.device = device
        self.knn = knn
    
    def __call__(self, batch):
        batch_fit, batch_supervision, batch_location = zip(*batch)
        batch_fit = torch.from_numpy(np.vstack(batch_fit)).to(self.device)
        if batch_supervision[0] is not None:
            batch_supervision = torch.from_numpy(np.vstack(batch_supervision)).to(self.device)
        batch_location = np.vstack(batch_location)
        batch_edge = location_to_edge(batch_location, self.knn).to(self.device)
        
        return batch_fit, batch_supervision, batch_edge


class rnaCollate():
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def __call__(self, batch):
        batch_fit, batch_supervision, batch_label = zip(*batch)
        batch_fit = torch.from_numpy(np.vstack(batch_fit)).to(self.device)
        batch_supervision = torch.from_numpy(np.vstack(batch_supervision)).to(self.device)
        batch_label = torch.from_numpy(np.vstack(batch_label)).long().squeeze(-1).to(self.device)
        batch_genegraph = cross_dist(batch_fit, batch_supervision)
        
        return batch_fit, batch_supervision, batch_label, batch_genegraph