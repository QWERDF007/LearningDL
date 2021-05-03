import argparse
import copy
from pathlib import Path

import cv2
import numpy as np

import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

from model import create_model

methods = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image', type=str, metavar='PATH',
                        help='path to image')
    parser.add_argument('--output', type=str, metavar='DIR',
                        help='output directory')
    parser.add_argument('--arch', type=str, metavar='NAME', default='resnet50',
                        help='name of model')
    parser.add_argument('--ckpt', type=str, metavar='PATH',
                        help='path to checkpoint')
    parser.add_argument('--num-classes', type=int, default=2, metavar='N',
                        help='number of classes')
    parser.add_argument('--target-layer', type=str, metavar='NAME',
                        help='target layer to visualize')
    parser.add_argument('--target', type=int, metavar='N',
                        help='target category to visualize')
    parser.add_argument('--method', type=str, metavar='NAME', default='scorecam', choices=methods.keys(),
                        help='visual method')
    parser.add_argument('--aug-smooth', action='store_true', default=False,
                        help='apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen-smooth', action='store_true', default=False,
                        help='reduce noise by taking the first principle componenet of cam_weights*activations')

    args = parser.parse_args()

    image_path = Path(args.image)
    model = create_model(args.arch, args.num_classes, pretrained=False)
    try:
        model.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=True)
    except:
        model.load_state_dict(torch.load(args.ckpt), strict=True)
    # print(model)
    output_dir = Path(args.output) if args.output else Path('./')
    target_category = args.target
    target_layer = getattr(model, args.target_layer)
    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=True)
    gb_model = GuidedBackpropReLUModel(model=copy.deepcopy(model), use_cuda=True)
    cam.batch_size = 32
    rgb_img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)[:, :, ::-1]
    # rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255.0
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=None,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    gb = gb_model(input_tensor, target_category=args.target)
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)
    cv2.imwrite(str(output_dir / (args.method + '_cam_' + image_path.name)), cam_image)
    cv2.imwrite(str(output_dir / ('gb_' + image_path.name)), gb)
    cv2.imwrite(str(output_dir / (args.method + '_cam_gb_' + image_path.name)), cam_gb)
