import os
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []

        # Wczytaj nazwy plików z katalogu
        self.file_names = os.listdir(data_dir)

        for file_name in self.file_names:
            file_path = os.path.join(data_dir, file_name)
            self.images.append((file_path, int(file_name.split('_')[1])))  # Zakładam, że numer klasy jest zawarty w nazwie pliku

    def __len__(self):
        return len(self.images)


