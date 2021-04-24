import os
import time
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import efficientnet_pytorch as efn
from efficientnet_pytorch import EfficientNet

models.resnet50()

MODEL_NAMES = [name for name in efn.VALID_MODELS]
MODEL_NAMES += [name for name in models.__dict__
                if name.islower() and not name.startswith("__")
                and callable(models.__dict__[name])]
MODEL_NAMES = sorted(MODEL_NAMES)


def create_model(name: str, num_classes, pretrained=True):
    if pretrained:
        if name.startswith('efficientnet'):
            model = EfficientNet.from_pretrained(name, num_classes=num_classes)
            return model
        else:
            model = models.__dict__[name](pretrained=True, num_classes=1000)
            if hasattr(model, 'classifier'):
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                raise NotImplementedError("This arch doesn't have pretrained model")
            return model
    else:
        if name.startswith('efficientnet'):
            model = EfficientNet.from_name(name, num_classes=num_classes)
            return model
        else:
            model = models.__dict__[name](pretrained=False, num_classes=num_classes)
            return model



def save_checkpoint(checkpoint, output_dir, epoch, loss, acc, is_best=False):
    now = time.strftime("%Y%m%d%H%M%S")
    fname = '{}-{}-{:.3f}-{:.2f}.pth.tar'.format(epoch, now, loss, acc)
    fpath = os.path.join(output_dir, fname)
    torch.save(checkpoint, fpath)
    if is_best:
        best_fpath = os.path.join(output_dir, 'model_best.pth.tar')
        shutil.copyfile(fpath, best_fpath)
