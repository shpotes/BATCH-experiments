import torch
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import functional as F
import pytorch_lightning as pl

def crop_grid(input_image, pad, size):
    _, height, width = input_image.shape

    return [
        [
            F.crop(input_image, h, w, size, size)
            for h in range(0, height, pad)
        ] for w in range(0, width, pad)
    ]

class BATCHDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=32, grid_size=512, grid_pad=None, val_porc=0.3, resize=128):
        super().__init__()

        self.root = root
        self.val_porc = val_porc
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.grid_pad = grid_size if grid_pad is None else grid_pad
        self.resize=resize
        self.preprocessing = transforms.Compose([
            transforms.Lambda(lambda x: crop_grid(x, self.grid_pad, self.grid_size)),
            transforms.Lambda(lambda grid: [[F.resize(x, self.resize) for x in row]for row in grid]),
            transforms.Lambda(lambda grid: torch.stack([item for sublist in grid for item in sublist])),
        ])


    def setup(self, stage=None):
        dataset = ImageFolder(
            self.root,
            transforms.Compose([
                transforms.ToTensor(),
                # uncomment if imagenet transfer learning.
                #transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406],
                #    std=[0.229, 0.224, 0.225]
                #),
                self.preprocessing,
                # transforms.Resize(self.resize)
            ])
        )

        val_size = int(len(dataset) * self.val_porc)
        train_size = len(dataset) - val_size

        self.train_dataset, self.val_dataset = data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )

def baseline_preprocessing(resize):
    return transforms.Resize((resize, resize))
