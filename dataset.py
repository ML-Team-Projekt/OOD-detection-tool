from torch.utils.data import Dataset
from utils import createCsv
import os
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        createCsv(self.rootDir)
        self.annotation =  pd.read_csv('output.csv')


    def __getitem__(self, index):
        data = self.annotation.iloc[index,0]
        label = self.annotation.iloc[index,1]
        return data, label

    def __len__(self):
        return len(self.annotation)



