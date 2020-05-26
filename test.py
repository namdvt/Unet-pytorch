import numpy as np
import tifffile
import torch
import matplotlib.pyplot as plt

from model import Model

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model = Model().to(device)
model.load_state_dict(torch.load('output/weight.pth', map_location=device))
model.eval()

if __name__ == '__main__':
    inputs = torch.tensor(np.expand_dims(tifffile.TiffFile('data/ISBI/test-volume.tif').asarray(), axis=1) / 255)
    with torch.no_grad():
        outputs = model(inputs.float().to(device))
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0

    for i in range(len(outputs)):
        fig = plt.figure()

        fig.add_subplot(1, 2, 1)
        plt.imshow(inputs[i].squeeze().cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Input')

        fig.add_subplot(1, 2, 2)
        plt.imshow(outputs[i].squeeze().cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('Output')

        plt.savefig('results/img_' + str(i) + '.png', dpi=150, bbox_inches='tight')
        plt.close()

    print('done')