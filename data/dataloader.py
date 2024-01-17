from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import os 
import torch
from torchvision import transforms

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = []
        if root[-3:] == 'txt':
            f = open(root, 'r')
            lines = f.readlines()            
            for line in lines:
                self.images.append(line.strip())
        else:
            self.images = sorted(os.listdir(self.root))
            self.images = [
                os.path.join(self.root, x) for x in self.images
            ]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        w, h = img.size
        nw, nh = (w // 256) * 256, (h // 256) * 256
        img = img.crop((0, 0, nw, nh))

        if self.transform is not None:
            return self.transform(img), self.images[index]
        else:
            return img, self.images[index]

    def __len__(self):
        return len(self.images)

