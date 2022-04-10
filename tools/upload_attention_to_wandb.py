import os
import json
import glob
import torch
import wandb
import cv2
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.extend(['.', '../'])

from visualize_attention import load_model_eval, get_self_attention_from_image, load_image_from_path


IMAGES_FOLDER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
IMAGES_METADATA_JSON = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data.json')
PRETRAINED_MODEL_PATH = os.path.expanduser('~/nanit/dino/dino_deitsmall8_pretrain.pth')
VIT_ARCH = 'vit_small'
PATCH_SIZE = 8
IMAGE_SIZE = (480, 480)

# Filtering - Uploading only RGB images


def get_image_from_input_tensor(input_tensor_to_model):
    input_tensor_to_model_scaled = torchvision.utils.make_grid(input_tensor_to_model, normalize=True, scale_each=True)
    image_np = np.uint8(255 * input_tensor_to_model_scaled.cpu().numpy().transpose(1, 2, 0))
    return Image.fromarray(image_np, 'RGB')


def get_attention_heatmap(attention_head):
    # heatmap = cv2.normalize(attention_head, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    # return heatmap

    heatmap_tmp_fp = 'attention_head.png'
    plt.imsave(fname=heatmap_tmp_fp, arr=attention_head, format='png')
    heatmap = Image.open(heatmap_tmp_fp).convert('RGB')
    return heatmap


def main():
    with open(IMAGES_METADATA_JSON, 'r') as f:
        images_metadata = json.load(f)

    images_paths = glob.glob(os.path.join(IMAGES_FOLDER, '*'))
    images_paths = [p for p in images_paths if p.endswith('png') or p.endswith('jpg')]

    model, model_device = load_model_eval(PRETRAINED_MODEL_PATH, VIT_ARCH, PATCH_SIZE)

    columns = ['filename', 'image', 'input_image', 'pose', 'attn-head0', 'attn-head1', 'attn-head2', 'attn-head3', 'attn-head4', 'attn-head5']
    attentions_map_table = wandb.Table(columns=columns)

    images_metadata_filtered = {}
    for k, v in tqdm(images_metadata.items(), desc='Images Filtering'):
        if v['is_ir']:
            continue
        image_path = os.path.join(IMAGES_FOLDER, k)
        if image_path not in images_paths:
            print('image DOES NOT Exist', k)
            continue

        images_metadata_filtered[k] = v
    print('Images for Attention', len(images_metadata_filtered.keys()))

    with torch.no_grad():
        for k, v in tqdm(images_metadata_filtered.items(), desc='Images'):
            image_path = os.path.join(IMAGES_FOLDER, k)
            img = load_image_from_path(image_path)
            input_tensor_to_model, attentions, _, num_of_heads = get_self_attention_from_image(img, model, model_device, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE)

            input_img_to_model_rgb = get_image_from_input_tensor(input_tensor_to_model)
            # input_img_to_model_rgb.save('input_img_to_model_rgb.png')

            attn0 = get_attention_heatmap(attentions[0])
            attn1 = get_attention_heatmap(attentions[1])
            attn2 = get_attention_heatmap(attentions[2])
            attn3 = get_attention_heatmap(attentions[3])
            attn4 = get_attention_heatmap(attentions[4])
            attn5 = get_attention_heatmap(attentions[5])

            attentions_map_table.add_data(k, wandb.Image(img), wandb.Image(input_img_to_model_rgb), v['pose'],
                                          wandb.Image(attn0), wandb.Image(attn1), wandb.Image(attn2),
                                          wandb.Image(attn3), wandb.Image(attn4), wandb.Image(attn5))

    wandb.init(project='self-supervised-exploration', entity='algo', job_type='eval-attn-maps')
    wandb.log({'Attention Maps Table': attentions_map_table})


if __name__ == '__main__':
    main()
