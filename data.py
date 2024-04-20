import os

from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data/'):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        datasets.SVHN(root=self.data_dir)
        pass