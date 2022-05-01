import torch
import utils
import vision_transformer as vits

from PIL import Image

VIT_ARCH = 'vit_small'
PATCH_SIZE = 8


def load_model_eval(pretrained_weights, arch=VIT_ARCH, patch_size=PATCH_SIZE, checkpoint_key='teacher'):
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
