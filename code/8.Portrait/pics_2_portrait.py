import argparse
from tqdm import tqdm
from pathlib import Path

import cv2
import torch
import numpy as np
from U_2_Net.model import U2NET
import warnings

warnings.filterwarnings('ignore')


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def main(args):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(args.ckpt))
    if args.use_cuda:
        net.cuda()
    net.eval()
    paths = list(Path(args.data).iterdir())
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    with torch.no_grad():
        for path in tqdm(paths, ncols=100, unit='files'):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            tmpImg = np.zeros((img.shape[0], img.shape[1], 3))
            # normalize and bgr to rgb
            img = img / np.max(img)
            tmpImg[:, :, 2] = (img[:, :, 0] - 0.406) / 0.225
            tmpImg[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 0] = (img[:, :, 2] - 0.485) / 0.229
            tmpImg = tmpImg.transpose((2, 0, 1))
            tmpImg = tmpImg[np.newaxis, :, :, :]
            tmpImg = torch.from_numpy(tmpImg)
            tmpImg = tmpImg.type(torch.FloatTensor)
            if args.use_cuda:
                tmpImg = tmpImg.cuda()
            d1, d2, d3, d4, d5, d6, d7 = net(tmpImg)
            pred = 1.0 - d1[:, 0, :, :]
            pred = normPRED(pred)
            pred = pred.squeeze()
            pred = pred.cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            # pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(output_dir / path.name), pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, metavar='DIR', help='input directory')
    parser.add_argument('--ckpt', type=str, metavar='PATH', help='path to checkpoint')
    parser.add_argument('--output', type=str, metavar='DIR', help='output directory')
    parser.add_argument('--use-cuda', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    main(args)
