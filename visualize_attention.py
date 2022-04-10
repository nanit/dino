# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def get_input_transforms(image_size):
    return pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def get_self_attention_from_image(_img, _model, device, patch_size=8, image_size=(480, 480), transform=None, threshold=None):
    if transform is None:
        transform = get_input_transforms(image_size)

    _input_tensor_to_model = transform(_img)

    # make the image divisible by the patch size
    w, h = _input_tensor_to_model.shape[1] - _input_tensor_to_model.shape[1] % patch_size, _input_tensor_to_model.shape[2] - _input_tensor_to_model.shape[2] % patch_size
    _input_tensor_to_model = _input_tensor_to_model[:, :w, :h].unsqueeze(0)

    w_featmap = _input_tensor_to_model.shape[-2] // patch_size
    h_featmap = _input_tensor_to_model.shape[-1] // patch_size

    _attentions = _model.get_last_selfattention(_input_tensor_to_model.to(device))

    _nh = _attentions.shape[1]   # number of head

    # we keep only the output patch attention
    _attentions = _attentions[0, :, 0, 1:].reshape(_nh, -1)

    _th_attn = None
    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(_attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        _th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(_nh):
            _th_attn[head] = _th_attn[head][idx2[head]]
        _th_attn = _th_attn.reshape(_nh, w_featmap, h_featmap).float()
        # interpolate
        _th_attn = nn.functional.interpolate(_th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    _attentions = _attentions.reshape(_nh, w_featmap, h_featmap)
    _attentions = nn.functional.interpolate(_attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return _input_tensor_to_model, _attentions, _th_attn, _nh


def load_model_eval(pretrained_weights, arch, patch_size, checkpoint_key='teacher'):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    _model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    for p in _model.parameters():
        p.requires_grad = False
    _model.eval()
    _model.to(device)
    utils.load_pretrained_weights(_model, pretrained_weights, checkpoint_key, arch, patch_size, device=device)

    return _model, device


def load_image_from_path(image_path):
    with open(image_path, 'rb') as fid:
        _img = Image.open(fid)
        _img = _img.convert('RGB')
    return _img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    model, model_device = load_model_eval(args.pretrained_weights, args.arch, args.patch_size)

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        img = load_image_from_path(args.image_path)
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)

    input_tensor_to_model, attentions, th_attn, num_of_heads = get_self_attention_from_image(img, model, model_device,
                                                                                             patch_size=args.patch_size, image_size=args.image_size,
                                                                                             threshold=args.threshold)

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(input_tensor_to_model, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(num_of_heads):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(num_of_heads):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
