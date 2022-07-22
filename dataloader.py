import torch
import json 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BartTokenizer


class MReDDataset(Dataset):
    def __init__(self,file_path):
        with open(file_path,'r') as f:
            data = json.load(f)
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        article = self.data[idx]['article']
        abstract = self.data[idx]['summary']
        ap = self.data[idx]['aspect_id']
        return article,abstract,ap