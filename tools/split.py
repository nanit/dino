import os
import json
import glob
from tqdm import tqdm

from python_tools.Split import BaseSplitModule
from python_tools import Constants, S3Handler

IMAGES_FOLDER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
IMAGES_METADATA_JSON = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data.json')
UPLOAD_TO_S3 = True


class DinoSplit(BaseSplitModule):
    def __init__(self, db_json_path: str, upload_to_s3: bool = False):
        super(DinoSplit, self).__init__(db_json_path=db_json_path, task='dino', upload_to_s3=upload_to_s3)

        self.pose_class = {}
        self.baby_predictions = {}
        self.head_predictions = {}

        self.images_folder = IMAGES_FOLDER

    def load_annotations_from_db_json(self):
        for file_name, file_dict in self.db_data.items():
            self.pose_class[file_name] = file_dict['pose']
            self.head_predictions[file_name] = file_dict['head']
            self.baby_predictions[file_name] = file_dict['baby']

    def build_meta_data_list(self):
        meta_data = []
        for i in tqdm(range(self.num_files), desc='building metadata list'):
            image_relative_path = self.images_paths[i]
            if image_relative_path[0] == '/':
                image_relative_path = image_relative_path[1:]
            baby_uid = self.db_data[image_relative_path]['baby_uid']
            image_path = os.path.join(self.images_folder, image_relative_path)
            file_name = self.images_paths[i]
            meta_data.append(self.create_image_metadata(file_name, image_path, baby_uid))
        return meta_data

    def create_image_metadata(self, file_name, image_path, baby_uid):
        image_metadata = None
        try:            # in-case some images don't have skeleton prediction and file_name is not in self.joints_predictions / self.joints_confidence
            image_metadata = {
                'file_name': file_name,
                'path': image_path,
                'baby_uid': baby_uid,
                'pose': self.pose_class[file_name],
                'head': self.head_predictions[file_name],
                'baby': self.baby_predictions[file_name]
            }
        except KeyError:
            pass
        return image_metadata

    def copy_images_to_s3(self, datasets_data, images_folder='images'):
        target_directory = self.get_s3_target_directory()
        source_paths_file_list = []
        destination_paths_file_list = []
        for dataset_files in datasets_data.values():
            for file_dict in tqdm(dataset_files, desc='copy images to S3'):
                source_path = file_dict['path']
                new_path = '/'.join((target_directory, images_folder, os.path.basename(source_path)))
                source_paths_file_list.append(source_path)
                destination_paths_file_list.append(new_path)
                file_dict['path'] = new_path

        if self.upload_to_s3:
            sub_list_len = 200
            for i in range(0, len(source_paths_file_list), sub_list_len):  # TODO divide to sublist in python_tools.S3Handler
                source_paths_file_sub_list = source_paths_file_list[i:i + sub_list_len]
                destination_paths_file_sub_list = destination_paths_file_list[i:i + sub_list_len]
                S3Handler.upload_file_list(source_paths_file_sub_list, destination_paths_file_sub_list, Constants.ALGORITHMS_BUCKET)

    def generate_histograms(self, datasets):
        """
        wrapper in case you want to add additional stats
        BaseSplitModule.build_dataset_plots, includes
         - ir/rgb
         - camera gen
        """
        additional_stats_mapping_dicts_list = [
            {
                'statistics': {},                       # pass an empty dict
                'mapping': self.pose_class,             # filename to value mapping
                'plot_title': 'Pose Classes',           # Plot Title
                'plot_filename': 'pose_classes.png'     # Plot filename for saving
             }
        ]

        self.build_dataset_plots(datasets, './histograms', additional_stats_mapping_dicts_list)

    def run(self):
        self.load_annotations_from_db_json()
        self.divide_babies_to_train_val_test(class_key_for_stratified='pose')
        self.print_babies_division()

        datasets = self.process_images_to_dataset()
        self.copy_images_to_s3(datasets)

        self.generate_histograms(datasets)

        self.save_and_upload_split(datasets)


def filter_images(images_metadata, images_paths):
    images_metadata_filtered = {}
    for k, v in tqdm(images_metadata.items(), desc='Images Filtering'):
        if v['is_ir']:
            continue
        image_path = os.path.join(IMAGES_FOLDER, k)
        if image_path not in images_paths:
            print('image DOES NOT Exist', k)
            continue

        images_metadata_filtered[k] = v
    return images_metadata_filtered


def main():
    with open(IMAGES_METADATA_JSON, 'r') as f:
        images_metadata = json.load(f)

    images_paths = glob.glob(os.path.join(IMAGES_FOLDER, '*'))
    images_paths = [p for p in images_paths if p.endswith('png') or p.endswith('jpg')]

    images_metadata_filtered_path = os.path.splitext(IMAGES_METADATA_JSON)[0] + '_filter' + os.path.splitext(IMAGES_METADATA_JSON)[1]
    if not os.path.exists(images_metadata_filtered_path):
        images_metadata_filtered = filter_images(images_metadata, images_paths)
        with open(images_metadata_filtered_path, 'w') as f:
            json.dump(images_metadata_filtered, f)
        print('Saved Images Filtered: {}'.format(images_metadata_filtered_path))
    else:
        print('Loaded Images Filtered: {}'.format(images_metadata_filtered_path))

    dino_split_module = DinoSplit(db_json_path=images_metadata_filtered_path, upload_to_s3=UPLOAD_TO_S3)
    dino_split_module.run()
    print('Finished')


if __name__ == '__main__':
    main()
