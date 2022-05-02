import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms

from dino_lib.config.constants import IMAGE_SIZE, PATCH_SIZE


def get_input_transforms(image_size):
    return pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def get_input_tensor_to_model(_img, patch_size=8, image_size=IMAGE_SIZE, transform=None):
    if transform is None:
        transform = get_input_transforms(image_size)
    _input_tensor_to_model = transform(_img)
    # make the image divisible by the patch size
    w, h = _input_tensor_to_model.shape[1] - _input_tensor_to_model.shape[1] % patch_size, _input_tensor_to_model.shape[2] - _input_tensor_to_model.shape[2] % patch_size
    _input_tensor_to_model = _input_tensor_to_model[:, :w, :h]
    return _input_tensor_to_model


def get_self_attention_from_image(_img, _model, device, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE, transform=None, threshold=None):
    _input_tensor_to_model = get_input_tensor_to_model(_img, patch_size, image_size, transform).unsqueeze(0)

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


def get_embedding_and_selfattention_from_batch(input_tensor_to_model, model, device, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE, transform=None, threshold=None):
    w_featmap = input_tensor_to_model.shape[-2] // patch_size
    h_featmap = input_tensor_to_model.shape[-1] // patch_size

    embedding, attentions = model.forward_and_last_selfattention(input_tensor_to_model.to(device))

    batch_size, num_of_heads = attentions.shape[:2]   # number of head

    # we keep only the output patch attention
    # # # # DON'T SUPPORT IN THRESHOLDS # # # #

    attentions = attentions[:, :, 0, 1:].reshape(batch_size, num_of_heads, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions, scale_factor=patch_size, mode="nearest").cpu().numpy()

    return embedding, attentions
