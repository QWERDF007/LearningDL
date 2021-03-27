import os
import gzip
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam, lr_scheduler


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


class MyDataset(Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            Y = self.target_transform(Y)
        return X, Y


def extract_data(data_dir, x_name, y_name):
    x_path = os.path.join(data_dir, x_name)
    y_path = os.path.join(data_dir, y_name)

    with gzip.open(x_path, 'rb') as f:
        X_content = f.read()
    with gzip.open(y_path, 'rb') as f:
        Y_content = f.read()

    X_bytes = len(X_content)
    Y_bytes = len(Y_content)

    height = int(X_content[8:12].hex(), 16)
    width = int(X_content[12:16].hex(), 16)
    pixels = height * width

    X = []
    Y = []

    for i in range(16, X_bytes, pixels):
        X.append(np.frombuffer(X_content[i:i + pixels], dtype=np.uint8).reshape(28, 28, 1))
    for i in range(8, Y_bytes):
        Y.append(Y_content[i])

    X = np.array(X)
    Y = np.array(Y, dtype=np.int64)

    return X, Y


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    mnist_dir = "./data"  # 下载的MNIST的文件夹路径
    X_train, Y_train = extract_data(mnist_dir, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    X_test, Y_test = extract_data(mnist_dir, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=123)
    print(X_train.shape, X_val.shape)
    print(Y_train.shape, Y_val.shape)

    net = Net()
    print(net)

    transform = transforms.ToTensor()

    datasets = {
        'train': MyDataset(X_train, Y_train, transform),
        'val': MyDataset(X_val, Y_val, transform)
    }

    dataset_sizes = {
        'train': len(datasets['train']),
        'val': len(datasets['val'])
    }

    batch_size = 512
    workers = 4

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True),
        'val': DataLoader(datasets['val'], batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 10

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    test_set = MyDataset(X_test, Y_test, transform)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=workers, pin_memory=True)
    total = len(test_set)
    correct = 0

    model.load_state_dict(best_model_wts)
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

    print()
    print("Test Acc: {:.4f}".format(correct.double() / total))

    path = "mnist.pt"
    torch.save(best_model_wts, path)
