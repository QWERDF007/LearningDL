import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder


class ImageDataset(Dataset):
    VALID_EXTENSION = ['*.jpg', '*.png']
    _repr_indent = 4

    def __init__(self, root, df: pd.DataFrame = None, transforms=None, target_transforms=None, multi=False,
                 evaluate=False):
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.multi = multi
        if multi and df is not None:
            self.columns = df.columns
            self.paths = [os.path.join(root, name) for name in df.values[:, 0]]
            if not evaluate:
                self.targets = np.array(df.values[:, 1:], dtype=np.float32)
            else:
                self.targets = [name for name in df.values[:, 0]]
        elif not multi:
            if df is not None:
                self.paths = [os.path.join(root, name) for name in df.values[:, 0]]
                if not evaluate:
                    self.targets = np.array(df.values[:, 1], dtype=np.int32)
                else:
                    self.targets = [name for name in df.values[:, 0]]
            else:
                paths = []
                for ext in self.VALID_EXTENSION:
                    paths += [path for path in Path(root).rglob(ext)]
                if not evaluate:
                    self.targets = [int(path.parent.stem) for path in paths]
                else:
                    self.targets = [path.stem for path in paths]
                self.paths = [str(path) for path in paths]
        else:
            raise NotImplementedError("No implement")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        target = self.targets[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return img, target

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["multi: {}".format(self.multi)]
        body += ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += ["Transforms: {}".format(repr(self.transforms))]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""


def create_dataset(root, df=None, mean=None, std=None, multi=False, train=True, evaluate=False) -> ImageDataset:
    transform = []
    if train:
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    if mean is not None and std is not None:
        transform.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform)
    dataset = ImageDataset(root, df, transforms=transform, multi=multi, evaluate=evaluate)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, metavar='DIR', help='directory to data')
    parser.add_argument('--csv', type=str, metavar='PATH', help='file to training annotation file')
    parser.add_argument('--vcsv', type=str, metavar='PATH', help='file to validation annotation file')
    parser.add_argument('--multi', action='store_true', default=False, help='multiple label')
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--no-use', action='store_true', default=False, help='no use')
    args = parser.parse_args()
    # pprint(vars(args))
    df = None
    if args.csv is not None:
        df = pd.read_csv(args.csv)
    fit = df[df.columns.values[1:]].apply(LabelEncoder().fit_transform)
    df[df.columns.values[1:]] = fit
    dataset = create_dataset(args.data, df, args.mean, args.std, args.multi, args.evaluate)
    print(dataset)
