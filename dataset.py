from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms


class SegmentationData(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.folder = root
        self.indexes = open(root + '/annotations_small.txt').read().splitlines()
        self.transform = transform

    def __getitem__(self, index):
        input_image, target_mask = self.indexes[index].split('\t')

        input_image = Image.open(input_image).convert('RGB')
        target_mask = Image.open(target_mask).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            target_mask = self.transform(target_mask)

        return input_image, target_mask

    def __len__(self):
        return len(self.indexes)


def get_loader(root, batch_size, shuffle=True, transform=None):
    dataset = SegmentationData(root=root, transform=transform)

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


# if __name__ == '__main__':
#     transform = transforms.Compose([transforms.Resize([128, 128]),
#                                     transforms.ToTensor()])
#     train_loader, val_loader = get_loader('data/', batch_size=2, transform=transform, shuffle=True)
#     for input_image, target_mask in train_loader:
#         print()
#     print()