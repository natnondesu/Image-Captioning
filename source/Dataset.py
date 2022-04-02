import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImgCaption_Dataset(Dataset):
    def __init__(self, img_path, caption_tok, transform=None):
        self.img_path = img_path
        self.y = torch.IntTensor(caption_tok)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        x = self.transform(img)
        y = self.y[index] # Retrieve data
        return x, y

    def __len__(self):
        return self.y.shape[0]