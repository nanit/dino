import os
import json
import torch
import wandb
import numpy as np
from tqdm import tqdm
import sys
sys.path.extend(['.', '../'])

from visualize_attention import load_model_eval, load_image_from_path, get_input_tensor_to_model

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

    features = torch.zeros((n_images, model.embed_dim)).to(model_device)
    labels = torch.zeros((n_images, )).to(model_device)
    with torch.no_grad():
        for i, (k, v) in enumerate(tqdm(images_metadata_filtered.items(), desc='Images')):
            image_path = os.path.join(IMAGES_FOLDER, k)
            img = load_image_from_path(image_path)
            input_tensor_to_model = get_input_tensor_to_model(img, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE)
            features[i] = model(input_tensor_to_model.to(model_device))
            labels[i] = pose_mapping[v['pose']]

    features = torch.nn.functional.normalize(features, dim=1, p=2)
    labels = torch.tensor([s for s in labels]).long()

    features, labels = features.cpu(), labels.cpu()

    checkpoint_data = {
        'features': features,
        'labels': labels
    }
    torch.save(checkpoint_data, FEATURES_SAVE_PATH)
    print('Saved Features and Labels to', FEATURES_SAVE_PATH)

    return checkpoint_data['features'], checkpoint_data['labels']


@torch.no_grad()
def nanit_knn_classifier(features, labels, k, temperature, num_classes=6):
    """
    based on knn_classifier as in eval_knn.py
    classify by dataset on his own
    don't use exp_() in distance transform

    """
    correct_total, total = 0.0, 0
    train_features, train_labels = features.clone(), labels.clone()
    test_features, test_labels = features.clone(), labels.clone()

    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx: min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices).long()

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temperature)#.exp_()
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
    features, labels = extract_features()
    temperature = 0.07

    k_list = [1, 10, 20, 100, 200]

    for k in k_list:
        accuracy = nanit_knn_classifier(features, labels, k, temperature, num_classes=6)
        print(f"{k}-NN classifier result: {accuracy}")


if __name__ == '__main__':
    main()
