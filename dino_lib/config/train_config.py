import os


class TrainConfig:
    def __init__(self):
        self.images_folder = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
        self.split_path = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data_filter_split.pkl')  # Filtered - Uploading only RGB images
        self.gpu_ids = 0

        self.output_dir = 'output/'
        self.saveckp_freq = 10
        self.print_freq = 100

        self.batch_size_per_gpu = 8
        self.num_workers = 8
        self.epochs = 100
        self.warmup_epochs = 10

        self.lr = 0.0005
        self.min_lr = 1e-6
        self.clip_grad = 3.0
        self.freeze_last_layer = 1

        self.weight_decay = 0.04
        self.weight_decay_end = 0.4

        self.global_crops_scale = (0.4, 1.)
        self.local_crops_scale = (0.05, 0.4)
        self.local_crops_number = 8

        self.vit_arch = 'vit_small'
        self.patch_size = 8

        self.drop_path_rate = 0.1       # stochastic depth rate
        self.out_dim = 65536            # Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well
        self.use_bn_in_head = False
        self.norm_last_layer = False     # Whether or not to weight normalize the last layer of the DINO head.
                                         # Not normalizing leads to better performance but can make the training unstable.
                                         # In our experiments, we typically set this paramater to False with vit_small and True with vit_base

        self.warmup_teacher_temp = 0.04
        self.teacher_temp = 0.04        # Final value (after linear warmup)
                                        # of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
                                        # starting with the default value of 0.04 and increase this slightly if needed.

        self.warmup_teacher_temp_epochs = 30
        self.momentum_teacher = 0.996

        self.optimizer = 'adamw'       # ['adamw', 'sgd', 'lars']

        self.use_fp16 = True
