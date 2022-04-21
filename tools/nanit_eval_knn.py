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

from visualize_attention import load_model_eval, get_self_attention_from_image, load_image_from_path, get_input_tensor_to_model

from python_tools.Mappings import gen_pose_class_to_number_mapping

IMAGES_FOLDER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
IMAGES_METADATA_JSON_FILTER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data_filter.json')    # Filtered - Uploading only RGB images
FEATURES_SAVE_PATH = os.path.expanduser('~/nanit/dino_data/features_filter.pth')
PRETRAINED_MODEL_PATH = os.path.expanduser('~/nanit/dino/dino_deitsmall8_pretrain.pth')
VIT_ARCH = 'vit_small'
PATCH_SIZE = 8
IMAGE_SIZE = (480, 480)


def extract_features():
    if os.path.exists(FEATURES_SAVE_PATH):
        checkpoint_data = torch.load(FEATURES_SAVE_PATH)
        print('Load Features and Labels from', FEATURES_SAVE_PATH)
        return checkpoint_data['features'], checkpoint_data['labels']

    with open(IMAGES_METADATA_JSON_FILTER, 'r') as f:
        images_metadata_filtered = json.load(f)

    n_images = len(images_metadata_filtered.keys())
    print('Images for K-NN', n_images)

    model, model_device = load_model_eval(PRETRAINED_MODEL_PATH, VIT_ARCH, PATCH_SIZE)
    pose_mapping = gen_pose_class_to_number_mapping()

    embedding_all = torch.zeros((n_images, model.embed_dim)).to(model_device)
    labels = torch.zeros((n_images, )).to(model_device)
    with torch.no_grad():
        for i, (k, v) in enumerate(tqdm(images_metadata_filtered.items(), desc='Images')):
            image_path = os.path.join(IMAGES_FOLDER, k)
            img = load_image_from_path(image_path)
            input_tensor_to_model = get_input_tensor_to_model(img, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE)
            embedding_all[i] = model(input_tensor_to_model.to(model_device))
            labels[i] = pose_mapping[v['pose']]

    embedding_all, labels = embedding_all.cpu(), labels.cpu()

    checkpoint_data = {
        'features': embedding_all,
        'labels': labels
    }
    torch.save(checkpoint_data, FEATURES_SAVE_PATH)
    print('Saved Features and Labels to', FEATURES_SAVE_PATH)

    return checkpoint_data['features'], checkpoint_data['labels']


def main():
    features, labels = extract_features()
    print('Features and Labels Exist')


if __name__ == '__main__':
    main()
