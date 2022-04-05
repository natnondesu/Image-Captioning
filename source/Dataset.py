import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImgCaption_Dataset(Dataset):
    def __init__(self, img_path, caption, caption_length, transform=None):
        self.img_path = img_path
        self.y = torch.LongTensor(caption)
        self.caplen = caption_length
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        x = self.transform(img)
        y = self.y[index] # Retrieve data
        caplen = self.caplen[index]
        return x, y, caplen

    def __len__(self):
        return self.y.shape[0]