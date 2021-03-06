import os
import pickle
from torch.utils.data import Dataset

from dino_lib.lib_utils.transforms import get_input_tensor_to_model
from dino_lib.lib_utils.core_utils import load_image_from_path
from dino_lib.config.constants import PATCH_SIZE, IMAGE_SIZE

from python_tools.Mappings import gen_pose_class_to_number_mapping


class DinoDataset(Dataset):
    def __init__(self, root_folder, split_data_path, image_set, is_train, patch_size=PATCH_SIZE, image_size=IMAGE_SIZE, data_aug=None):
        self.root_folder = root_folder
        self.image_set = image_set
        self.is_train = is_train
        self.patch_size = patch_size
        self.image_size = image_size
        self.pose_class_to_number = gen_pose_class_to_number_mapping()
        self.data_aug = data_aug

        with open(split_data_path, 'rb') as f:
            self.data = pickle.load(f)[image_set]

    def __getitem__(self, idx):
        data_rec = self.data[idx]
        file_name = data_rec['file_name']
        image_path = os.path.join(self.root_folder, file_name)
        img = load_image_from_path(image_path)

        if self.data_aug is None:
            input_tensor_to_model = get_input_tensor_to_model(img, patch_size=self.patch_size, image_size=self.image_size)
            pose_number = self.pose_class_to_number[data_rec['pose']]
            return input_tensor_to_model, pose_number, file_name

        else:
            img_resize = img.resize(self.image_size)     # Resize to imagenet size
            return self.data_aug(img_resize)

    def __len__(self):
        return len(self.data)
