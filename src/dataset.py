import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import random

class DeepfakeDataset(Dataset):
    def __init__(self, csv_path, split='train', transform=None):
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'frame_path']
        label = self.data.loc[idx, 'label']
        video_id = self.data.loc[idx, 'video_id']
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32), video_id

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
