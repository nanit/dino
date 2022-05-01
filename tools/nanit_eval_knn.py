import os
import json
import pickle
import torch
import wandb
import numpy as np
from tqdm import tqdm
import sys
sys.path.extend(['.', '../'])

from dino_lib.lib_utils.core_utils import load_model_eval
from dino_lib.core.dataset import DinoDataset

IMAGES_FOLDER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
SPLIT_PATH = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data_filter_split.pkl')    # Filtered - Uploading only RGB images
FEATURES_SAVE_PATH = os.path.expanduser('~/nanit/dino_data/features_filter.pth')
PRETRAINED_MODEL_PATH = os.path.expanduser('~/nanit/dino/dino_deitsmall8_pretrain.pth')
DATASETS_TO_UPLOAD = ['train', 'val']
BATCH_SIZE = 32
NUM_WORKERS = 8


def extract_features():
    if os.path.exists(FEATURES_SAVE_PATH):
        checkpoint_data = torch.load(FEATURES_SAVE_PATH)
        print('Load Features and Labels from', FEATURES_SAVE_PATH)
        return checkpoint_data

    model, model_device = load_model_eval(PRETRAINED_MODEL_PATH)
    features_dict = {}
    labels_dict = {}
    for dataset_name in DATASETS_TO_UPLOAD:
        dataset = DinoDataset(IMAGES_FOLDER, SPLIT_PATH, dataset_name, is_train=False)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        n_images = len(dataset)
        print('Images for K-NN', n_images)
        features = torch.zeros((n_images, model.embed_dim)).to(model_device)
        labels = torch.zeros((n_images, )).to(model_device)
        with torch.no_grad():
            for i, (input_tensor_to_model, pose_classes) in enumerate(tqdm(data_loader, desc='{} (# Batches)'.format(dataset_name.capitalize()))):
                actual_batch_size = input_tensor_to_model.size(0)
                features[(i*BATCH_SIZE):(i*BATCH_SIZE)+actual_batch_size] = model(input_tensor_to_model.to(model_device))
                labels[(i*BATCH_SIZE):(i*BATCH_SIZE)+actual_batch_size] = pose_classes

        features = torch.nn.functional.normalize(features, dim=1, p=2)
        labels = torch.tensor([s for s in labels]).long()

        features_dict[dataset_name], labels_dict[dataset_name] = features.cpu(), labels.cpu()

    checkpoint_data = {
        'features': features_dict,
        'labels': labels_dict
    }
    torch.save(checkpoint_data, FEATURES_SAVE_PATH)
    print('Saved Features and Labels to', FEATURES_SAVE_PATH)

    return checkpoint_data


@torch.no_grad()
def nanit_knn_classifier(checkpoint_data, k, temperature, num_classes=6):
    """
    based on knn_classifier as in eval_knn.py
    """
    correct_total, total = 0.0, 0
    train_features, train_labels = checkpoint_data['features']['train'], checkpoint_data['labels']['train']
    val_features, val_labels = checkpoint_data['features']['val'], checkpoint_data['labels']['val']

    train_features = train_features.t()
    num_test_images, num_chunks = val_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = val_features[idx: min((idx + imgs_per_chunk), num_test_images), :]
        targets = val_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices).long()

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temperature).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        correct_total += correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)
    accuracy = correct_total * 100.0 / total
    return accuracy


def main():
    checkpoint_data = extract_features()

    k_list = [5, 10, 20, 50, 100]

    for k in k_list:
        accuracy = nanit_knn_classifier(checkpoint_data, k, temperature=0.07, num_classes=6)
        print(f"{k}-NN classifier result: {accuracy}")


if __name__ == '__main__':
    main()
