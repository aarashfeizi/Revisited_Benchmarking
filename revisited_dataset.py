from PIL import Image
from torch.utils.data import Dataset


class Revisited_Dataset(Dataset):
    def __init__(self, cfg, loader_func_name, length, transform=None):
        self.cfg = cfg
        self.loader_func = cfg[loader_func_name]
        self.length = length
        self.transfrom = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img_path = self.loader_func(self.cfg, item)
        img = Image.open(img_path)

        if self.transfrom is not None:
            img = self.transfrom(img)

        return img
