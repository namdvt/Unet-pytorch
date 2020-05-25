from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import tifffile


class ISBIDataset(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.images = np.expand_dims(tifffile.TiffFile('data/ISBI/train-volume.tif').asarray(), axis=3)
        self.targets = np.expand_dims(tifffile.TiffFile('data/ISBI/train-labels.tif').asarray(), axis=3)
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

    def __len__(self):
        return len(self.images)


def get_loader(batch_size, shuffle=True, transform=None):
    dataset = ISBIDataset(transform=transform)

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
