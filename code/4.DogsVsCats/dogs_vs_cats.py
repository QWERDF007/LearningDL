import copy
import random
import time
import math
from pathlib import Path
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50


class DogsCatsSet(Dataset):
    def __init__(self, paths, transform=None, target_transform=None):
        self.paths = paths
        labels = [path.name.split('.')[0] for path in self.paths]
        self.labels = [0 if label == 'cat' else 1 for label in labels]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = str(self.paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        epoch_since = time.time()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print(
            "Time: {} / {}".format(timedelta(seconds=time_elapsed), timedelta(seconds=time_elapsed / (epoch / epochs))))
        print()

    time_elapsed = time.time() - since
    print("Training complete in {}".format(timedelta(seconds=time_elapsed)))
    print("Best Val Acc: {:.4f}".format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


def visualize_preds(model, device, dataloader, num_images=32):
    class_names = {
        0: 'cat',
        1: 'dog'
    }
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(32, 24))
    ncols = 8
    nrows = math.ceil(num_images / ncols)

    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                images_so_far += 1
                ax = plt.subplot(nrows, ncols, images_so_far)
                ax.axis('off')
                idx = preds[i].item()
                ax.set_title('{}: {:.3f}'.format(class_names[idx], outputs[i][idx].item()))
                imshow(images.cpu().data[i])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.savefig('blog4-3.jpg')
                    plt.show()
                    return
        model.train(mode=was_training)
    plt.show()


if __name__ == '__main__':
    data_dir = Path('data')  # 存放猫猫狗狗的文件夹路径
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test1'
    train_list = list(train_dir.iterdir())
    test_list = list(test_dir.iterdir())
    print('train: {}, test: {}'.format(len(train_list), len(test_list)))

    random.shuffle(train_list)
    total = len(train_list)
    val_size = int(total * 0.2)  # 训练验证8:2
    val_set = train_list[:val_size]
    train_set = train_list[val_size:]
    batch_size = 32
    workers = 6

    transform = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    datasets = {
        'train': DogsCatsSet(train_set, transform['train']),
        'val': DogsCatsSet(val_set, transform['val'])
    }

    dataset_sizes = {
        'train': len(datasets['train']),
        'val': len(datasets['val'])
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=workers,
                            pin_memory=True),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=True, pin_memory=True)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ft = resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_tf = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model_tf = train(model_ft, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, device, 20)

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_set = DogsCatsSet(test_list, test_transform)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    visualize_preds(model_ft, device, test_dataloader, 64)
