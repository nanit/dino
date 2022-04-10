import os
import glob
import pickle
import json
import cv2
from tqdm import tqdm

from python_tools.BBox import BBox
from python_tools.OSUtils import ensure_dir

IMAGES_FOLDER = os.path.expanduser('~/nanit/pose-classification/data/pose_190K/images/')
OUTPUT_FOLDER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
OUTPUT_JSON = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data.json')
DB_DATA_PATH = os.path.expanduser('~/nanit/pose_data/pose_190K/pose_2022_04_03_db_data/db_data.json')
# DETECTION_PREDICTIONS_PATH = os.path.expanduser('~/nanit/pose_data/pose_190K/skeleton_2022_04_03/detection_predictions.pkl')
ADD_FOR_CROP = 0.05     # percentage for width and height


def enlarge_bbox_for_crop(bbox_xywh, add_for_crop):
    x_min, y_min, w, h = bbox_xywh
    x_min_new = max(0, x_min - ((add_for_crop / 2) * w))
    y_min_new = max(0, y_min - ((add_for_crop / 2) * h))
    w_new = (1 + add_for_crop) * w
    h_new = (1 + add_for_crop) * h

    return [x_min_new, y_min_new, w_new, h_new]


def main():
    ensure_dir(OUTPUT_FOLDER)
    ensure_dir(os.path.dirname(OUTPUT_JSON))
    image_paths = glob.glob(os.path.join(IMAGES_FOLDER, '**'), recursive=True)
    image_paths = [p for p in image_paths if p.endswith('.png') or p.endswith('.jpg')]
    print('Found', len(image_paths), 'Images in', IMAGES_FOLDER)

    with open(DB_DATA_PATH, 'r') as f:
        db_data = json.load(f)
    print('Read', DB_DATA_PATH)

    # with open(DETECTION_PREDICTIONS_PATH, 'rb') as f:
    #     detection_predictions = pickle.load(f)
    # print('Read', DETECTION_PREDICTIONS_PATH)

    db_data_filtered = {}
    for k, v in tqdm(db_data.items(), desc='Images Filtering'):
        image_path = os.path.join(IMAGES_FOLDER, k)
        if not (v['body_status'] == 'valid' and v['head_status'] == 'valid' and image_path in image_paths):
            continue
        if tuple(v['resolution']) not in [(1280, 960), (1920, 1080), (960, 1280), (1080, 1920), (1440, 1080), (1080, 1440)]:
            continue
        db_data_filtered[k] = v
    print('Images for Crop', len(db_data_filtered.keys()))

    crop_images_data = {}
    for k, v in tqdm(db_data_filtered.items(), desc='Images Cropping'):
        image_path = os.path.join(IMAGES_FOLDER, k)
        image = cv2.imread(image_path)
        baby_xywh_for_crop = enlarge_bbox_for_crop(v['body_mark'], ADD_FOR_CROP)
        baby_xyxy_for_crop = BBox.create_bbox_from_xywh_list(baby_xywh_for_crop)
        x1, y1, x2, y2 = [int(c) for c in baby_xyxy_for_crop.coordinates_to_list('xyxy')]
        crop_image = image[y1:y2, x1:x2].copy()
        new_k = os.path.splitext(k)[0] + '_crop' + os.path.splitext(k)[1]
        crop_image_path = os.path.join(OUTPUT_FOLDER, new_k)
        cv2.imwrite(crop_image_path, crop_image)
        baby_xywh_on_crop = [v['body_mark'][0] - x1, v['body_mark'][1] - y1, v['body_mark'][2], v['body_mark'][3]]
        head_xywh_on_crop = [v['head_mark'][0] - x1, v['head_mark'][1] - y1, v['head_mark'][2], v['head_mark'][3]]

        crop_images_data[new_k] = {
            'baby': baby_xywh_on_crop,
            'head': head_xywh_on_crop,
            'pose': v['baby_pose_mark'],
            'orig_resolution': v['resolution'],
            'is_ir': v['is_ir'],
            'baby_uid': v['baby_uid']
        }

    print('Saved', len(crop_images_data.keys()), 'Images')
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(crop_images_data, f)
    print('Saved', OUTPUT_JSON)


if __name__ == '__main__':
    main()
