from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import tifffile
import torchvision.transforms.functional as F
import random


class ISBIDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.images = np.expand_dims(tifffile.TiffFile('data/ISBI/train-volume.tif').asarray(), axis=3)
        self.targets = np.expand_dims(tifffile.TiffFile('data/ISBI/train-labels.tif').asarray(), axis=3)

    def __getitem__(self, index):
        image = F.to_pil_image(self.images[index])
        target = F.to_pil_image(self.targets[index])

        # augmentation
        rotate = random.randint(0, 3)
        if rotate == 1:
            image = F.rotate(image, 180, fill=(0,))
            target = F.rotate(target, 180, fill=(0,))
        if rotate == 2:
            image = F.rotate(image, 90, fill=(0,))
            target = F.rotate(target, 90, fill=(0,))
        if rotate == 3:
            image = F.rotate(image, -90, fill=(0,))
            target = F.rotate(target, -90, fill=(0,))

        flip = random.randint(0, 2)
        if flip == 1:
            image = F.hflip(image)
            target = F.hflip(target)
        if flip == 2:
            image = F.vflip(image)
            target = F.vflip(target)

        # image = F.resize(image, [64, 64])
        # target = F.resize(target, [64, 64])

        image = F.to_tensor(image)
        target = F.to_tensor(target)

        return image, target

    def __len__(self):
        return len(self.images)


def get_loader(batch_size, shuffle=True):
    dataset = ISBIDataset()

    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=True)

    return train_loader, val_loader
