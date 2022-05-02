import os
import torch
import wandb
import cv2
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
sys.path.extend(['.', '../'])

from dino_lib.lib_utils.transforms import get_embedding_and_selfattention_from_batch
from dino_lib.lib_utils.core_utils import load_model_eval, load_image_from_path
from dino_lib.core.dataset import DinoDataset

from python_tools.Mappings import gen_pose_number_to_pose_class_mapping


IMAGES_FOLDER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
SPLIT_PATH = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data_filter_split.pkl')
PRETRAINED_MODEL_PATH = os.path.expanduser('~/nanit/dino/dino_deitsmall8_pretrain.pth')
CHECKPOINT_PATH = os.path.expanduser('~/nanit/dino/output/checkpoint_train_02052022.pth')
DATASETS_TO_UPLOAD = ['train', 'val']
BATCH_SIZE = 16
NUM_WORKERS = 8
USE_CHECKPOINT = True

# Filtering - Uploading only RGB images


def get_image_from_input_tensor(input_tensor_to_model):
    images = []
    for i in range(input_tensor_to_model.size(0)):
        input_tensor_to_model_scaled = torchvision.utils.make_grid(input_tensor_to_model[i], normalize=True, scale_each=True)
        image_np = np.uint8(255 * input_tensor_to_model_scaled.cpu().numpy().transpose(1, 2, 0))
        images.append(Image.fromarray(image_np, 'RGB'))
    return images


def get_attention_heatmaps(attentions):
    attention_heatmaps = []
    for i in range(attentions.shape[0]):
        attention_heatmaps.append([get_attention_heatmap(attentions[i, 0]),
                                   get_attention_heatmap(attentions[i, 1]),
                                   get_attention_heatmap(attentions[i, 2]),
                                   get_attention_heatmap(attentions[i, 3]),
                                   get_attention_heatmap(attentions[i, 4]),
                                   get_attention_heatmap(attentions[i, 5])
                                   ])
    return attention_heatmaps


def get_attention_heatmap(attention_head):
    heatmap = cv2.normalize(attention_head, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def get_table_row(i, filenames, input_img_to_model_rgb, pose_classes, attention_heatmaps, embedding):
    pose_number_to_pose_class = gen_pose_number_to_pose_class_mapping()
    embedding_np = embedding[i].detach().cpu().numpy().reshape(-1)
    filename = filenames[i]
    image_path = os.path.join(IMAGES_FOLDER, filename)
    img = load_image_from_path(image_path)
    table_data_row = [filename, wandb.Image(img), wandb.Image(input_img_to_model_rgb[i]), pose_number_to_pose_class[int(pose_classes[i].detach().cpu())]]
    table_data_row.extend([wandb.Image(attn) for attn in attention_heatmaps[i]])
    table_data_row.append(embedding_np)
    return table_data_row


def main():
    if USE_CHECKPOINT:
        model, model_device = load_model_eval(CHECKPOINT_PATH, checkpoint_key='student')
        print('Loaded Model {}'.format(CHECKPOINT_PATH))
    else:
        model, model_device = load_model_eval(PRETRAINED_MODEL_PATH)
        print('Loaded Model {}'.format(PRETRAINED_MODEL_PATH))

    wandb.Table._MAX_EMBEDDING_DIMENSIONS = 1000  # for uploading np.array for embedding larger than default size
    columns = ['filename', 'image', 'input_image', 'pose', 'attn-head0', 'attn-head1', 'attn-head2', 'attn-head3', 'attn-head4', 'attn-head5', 'embedding']

    attentions_map_table_dict = {}
    for dataset_name in DATASETS_TO_UPLOAD:
        dataset = DinoDataset(IMAGES_FOLDER, SPLIT_PATH, dataset_name, is_train=False)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        table_data = []
        with torch.no_grad():
            for input_tensor_to_model, pose_classes, filenames in tqdm(data_loader, desc='{} (# Batches)'.format(dataset_name.capitalize())):
                embedding, attentions = get_embedding_and_selfattention_from_batch(input_tensor_to_model, model, model_device)

                input_img_to_model_rgb = get_image_from_input_tensor(input_tensor_to_model)
                # input_img_to_model_rgb.save('input_img_to_model_rgb.png')
                attention_heatmaps = get_attention_heatmaps(attentions)

                for i in range(input_tensor_to_model.size(0)):
                    table_data_row = get_table_row(i, filenames, input_img_to_model_rgb, pose_classes, attention_heatmaps, embedding)
                    table_data.append(table_data_row)

            attentions_map_table_dict[dataset_name] = wandb.Table(columns=columns, data=table_data)

    wandb_attention_dict = {'Attention Maps ' + dataset_name.capitalize() + ' Table': table for dataset_name, table in attentions_map_table_dict.items()}
    wandb.init(project='self-supervised-exploration', entity='algo', job_type='eval-attn-maps')
    wandb.log(wandb_attention_dict)


if __name__ == '__main__':
    main()
