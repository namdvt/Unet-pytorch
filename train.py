import torch
import torch.optim as optim
from helper import write_log, write_figures
import numpy as np
from dataset import get_loader
from model import Model
from tqdm import tqdm
from loss import DiceLoss


def fit(epoch, model, optimizer, criterion, device, data_loader, phase='training'):
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

        loss = criterion(outputs, targets)
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
    num_epochs = 100
    learning_rate = 0.1

    train_loader, val_loader = get_loader(batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = DiceLoss(smooth=1.)
    
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_epoch_loss = fit(epoch, model, optimizer, criterion, device, train_loader, phase='training')
        val_epoch_loss = fit(epoch, model, optimizer, criterion, device, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save(model.state_dict(), 'output/weight_sigmoid.pth')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_figures('output', train_losses, val_losses)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss)

        scheduler.step(val_epoch_loss)


if __name__ == "__main__":
    train()
