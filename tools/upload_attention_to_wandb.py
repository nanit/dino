import os
import json
import pickle
import torch
import wandb
import cv2
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
sys.path.extend(['.', '../'])

from visualize_attention import get_self_attention_from_image
from dino_lib.lib_utils.core_utils import load_model_eval, load_image_from_path

IMAGES_FOLDER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
SPLIT_PATH = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data_filter_split.pkl')
PRETRAINED_MODEL_PATH = os.path.expanduser('~/nanit/dino/dino_deitsmall8_pretrain.pth')
VIT_ARCH = 'vit_small'
PATCH_SIZE = 8
IMAGE_SIZE = (480, 480)
DATASETS_TO_UPLOAD = ['train', 'val']
# Filtering - Uploading only RGB images


def get_image_from_input_tensor(input_tensor_to_model):
    input_tensor_to_model_scaled = torchvision.utils.make_grid(input_tensor_to_model, normalize=True, scale_each=True)
    image_np = np.uint8(255 * input_tensor_to_model_scaled.cpu().numpy().transpose(1, 2, 0))
    return Image.fromarray(image_np, 'RGB')


def get_attention_heatmap(attention_head):
    heatmap = cv2.normalize(attention_head, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def main():
    with open(SPLIT_PATH, 'rb') as f:
        split_data = pickle.load(f)
    print('Read Split Data')

    model, model_device = load_model_eval(PRETRAINED_MODEL_PATH, VIT_ARCH, PATCH_SIZE)
    print('Loaded Model {}'.format(PRETRAINED_MODEL_PATH))

    wandb.Table._MAX_EMBEDDING_DIMENSIONS = 1000  # for uploading np.array for embedding larger than default size
    columns = ['filename', 'image', 'input_image', 'pose', 'attn-head0', 'attn-head1', 'attn-head2', 'attn-head3', 'attn-head4', 'attn-head5', 'embedding']

    attentions_map_table_dict = {}
    with torch.no_grad():
        for dataset_name in DATASETS_TO_UPLOAD:
            attentions_map_table_dict[dataset_name] = wandb.Table(columns=columns)
            data = split_data[dataset_name]
            for d in tqdm(data, desc='Images'):
                file_name = d['file_name']
                image_path = os.path.join(IMAGES_FOLDER, file_name)
                img = load_image_from_path(image_path)
                input_tensor_to_model, attentions, _, num_of_heads = get_self_attention_from_image(img, model, model_device, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE)
                embedding = model(input_tensor_to_model.to(model_device))
                embedding_np = embedding.detach().cpu().numpy().reshape(-1)

                input_img_to_model_rgb = get_image_from_input_tensor(input_tensor_to_model)
                # input_img_to_model_rgb.save('input_img_to_model_rgb.png')

                attn0 = get_attention_heatmap(attentions[0])
                attn1 = get_attention_heatmap(attentions[1])
                attn2 = get_attention_heatmap(attentions[2])
                attn3 = get_attention_heatmap(attentions[3])
                attn4 = get_attention_heatmap(attentions[4])
                attn5 = get_attention_heatmap(attentions[5])

                attentions_map_table_dict[dataset_name].add_data(file_name, wandb.Image(img), wandb.Image(input_img_to_model_rgb), d['pose'],
                                                                 wandb.Image(attn0), wandb.Image(attn1), wandb.Image(attn2),
                                                                 wandb.Image(attn3), wandb.Image(attn4), wandb.Image(attn5), embedding_np)

    wandb_attention_dict = {'Attention Maps ' + dataset_name.capitalize() + ' Table': table for dataset_name, table in attentions_map_table_dict.items()}
    wandb.init(project='self-supervised-exploration', entity='algo', job_type='eval-attn-maps')
    wandb.log(wandb_attention_dict)


if __name__ == '__main__':
    main()
