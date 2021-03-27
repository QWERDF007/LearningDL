import math
from pathlib import Path
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # N = (W - F + 2P) / S + 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # [32, 26, 26]
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # [64, 24, 24]
        # max_pool2d [64, 12, 12]
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # [128]
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)  # [10]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    data_dir = Path('data')
    checkpoint = '../../checkpoints/mnist.pt'
    net = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print(model)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    paths = list(data_dir.iterdir())
    total = len(paths)
    col = 4
    row = math.ceil(total / col)
    with torch.no_grad():
        for i, path in enumerate(paths, start=1):
            plt.subplot(row, col, i)
            img_path = str(path)
            img = Image.open(img_path).convert('L')
            img = img.point(lambda p: p < 128 and 255)
            tensor = transform(img)
            tensor = tensor.unsqueeze(0).to(device)
            output = model(tensor)
            preds = F.softmax(output, 1)
            v, idx = preds.topk(1)
            img = img.resize((28, 28))
            plt.imshow(img, cmap='gray')
            plt.title("{}: {:.3f}".format(idx.item(), v.item()))
    plt.show()
