import torch
import torch.optim as optim
from torchvision import transforms

from helper import write_log, write_figures
import numpy as np
from dataset import get_loader

from model import Model
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch.nn.functional as F


def fit(epoch, model, optimizer, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0

    for inputs, targets in tqdm(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            outputs = model(inputs)
        else:
            with torch.no_grad():
                outputs = model(inputs)

        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        running_loss += loss.item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
    return epoch_loss


def train():
    print('start training ...........')
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Model().to(device)
    batch_size = 2
    num_epochs = 200
    learning_rate = 0.01
    root = 'data/'

    transform = transforms.Compose([transforms.Resize([256, 256]),
                                    transforms.ToTensor()])

    train_loader, val_loader = get_loader(root, batch_size=batch_size, transform=transform, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)
    
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_epoch_loss = fit(epoch, model, optimizer, device, train_loader, phase='training')
        val_epoch_loss = fit(epoch, model, optimizer, device, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_figures('output', train_losses, val_losses)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss)

        scheduler.step(epoch)


if __name__ == "__main__":
    train()