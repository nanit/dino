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
DETECTION_PREDICTIONS_PATH = os.path.expanduser('~/nanit/pose_data/pose_190K/skeleton_2022_04_03/detection_predictions.pkl')
DETECTION_PREDICTIONS_FILTER_PATH = os.path.expanduser('~/nanit/dino_data/detections_predictions_filter.pkl')
ADD_FOR_CROP = 0.05     # percentage for width and height


def enlarge_bbox_for_crop(bbox_xyxy, add_for_crop):
    bbox = BBox(*bbox_xyxy)
    w_new = bbox.w * (1 + add_for_crop)
    h_new = bbox.h * (1 + add_for_crop)
    baby_bbox_center = bbox.get_bbox_center()
    baby_bbox_enlarge = BBox.create_bbox_from_center(baby_bbox_center[0], baby_bbox_center[1], w_new, h_new)

    return baby_bbox_enlarge.coordinates_to_list('xyxy')


def filter_detections_predictions(db_data, image_paths):
    if os.path.exists(DETECTION_PREDICTIONS_FILTER_PATH):
        with open(DETECTION_PREDICTIONS_FILTER_PATH, 'rb') as f:
            detection_predictions_filtered = pickle.load(f)
        print('Load Detection Prediction Filter', DETECTION_PREDICTIONS_FILTER_PATH)
    else:
        with open(DETECTION_PREDICTIONS_PATH, 'rb') as f:
            detection_predictions = pickle.load(f)
        print('Read', DETECTION_PREDICTIONS_PATH)

        detection_predictions_filtered = []
        for d in tqdm(detection_predictions, desc='Images Filtering'):
            key_in_db_data = os.path.basename(d['image_path'])
            image_path = os.path.join(IMAGES_FOLDER, key_in_db_data)
            if not (d['pred']['baby']['boxes'] and d['pred']['head']['boxes'] and image_path in image_paths):
                continue
            resolution = tuple(db_data[key_in_db_data]['resolution'])
            if resolution not in [(1280, 960), (1920, 1080), (960, 1280), (1080, 1920), (1440, 1080), (1080, 1440)]:
                continue
            detection_predictions_filtered.append(d)

        with open(DETECTION_PREDICTIONS_FILTER_PATH, 'wb') as f:
            pickle.dump(detection_predictions_filtered, f)
        print('Save Detection Prediction Filter', DETECTION_PREDICTIONS_FILTER_PATH)

    print('Images for Crop (Detection Predictions)', len(detection_predictions_filtered))
    return detection_predictions_filtered


def main():
    ensure_dir(OUTPUT_FOLDER)
    ensure_dir(os.path.dirname(OUTPUT_JSON))
    image_paths = glob.glob(os.path.join(IMAGES_FOLDER, '**'), recursive=True)
    image_paths = [p for p in image_paths if p.endswith('.png') or p.endswith('.jpg')]
    print('Found', len(image_paths), 'Images in', IMAGES_FOLDER)

    with open(DB_DATA_PATH, 'r') as f:
        db_data = json.load(f)
    print('Read', DB_DATA_PATH)

    detection_predictions_filtered = filter_detections_predictions(db_data, image_paths)

    crop_images_data = {}
    for d in tqdm(detection_predictions_filtered, desc='Images Cropping'):
        key_in_db_data = os.path.basename(d['image_path'])
        image_path = os.path.join(IMAGES_FOLDER, key_in_db_data)
        image = cv2.imread(image_path)

        baby_xyxy = d['pred']['baby']['boxes'][0]
        head_xyxy = d['pred']['head']['boxes'][0]
        baby_xyxy_for_crop = enlarge_bbox_for_crop(baby_xyxy, ADD_FOR_CROP)
        x1, y1, x2, y2 = [int(c) for c in baby_xyxy_for_crop]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, image.shape[1]), min(y2, image.shape[0])
        crop_image = image[y1:y2, x1:x2].copy()
        new_k = os.path.splitext(key_in_db_data)[0] + '_crop' + os.path.splitext(key_in_db_data)[1]
        crop_image_path = os.path.join(OUTPUT_FOLDER, new_k)
        cv2.imwrite(crop_image_path, crop_image)
        baby_xyxy_on_crop = [baby_xyxy[0] - x1, baby_xyxy[1] - y1, baby_xyxy[2] - x1, baby_xyxy[3] - y1]
        head_xyxy_on_crop = [head_xyxy[0] - x1, head_xyxy[1] - y1, head_xyxy[2] - x1, head_xyxy[3] - y1]

        crop_images_data[new_k] = {
            'baby': [float(c) for c in baby_xyxy_on_crop],
            'head': [float(c) for c in head_xyxy_on_crop],
            'orig_baby': [float(c) for c in baby_xyxy],
            'orig_head': [float(c) for c in head_xyxy],
            'pose': db_data[key_in_db_data]['baby_pose_mark'],
            'orig_resolution': db_data[key_in_db_data]['resolution'],
            'is_ir': db_data[key_in_db_data]['is_ir'],
            'baby_uid': db_data[key_in_db_data]['baby_uid'],
            'camera_id': db_data[key_in_db_data]['camera_id']
        }

    print('Saved', len(crop_images_data.keys()), 'Images')
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(crop_images_data, f)
    print('Saved', OUTPUT_JSON)


if __name__ == '__main__':
    main()
