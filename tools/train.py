# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.extend(['.', '..'])

import os
import datetime
import time
import json
from pathlib import Path

import torch
import torch.nn as nn

import utils
import vision_transformer as vits
from vision_transformer import DINOHead


from dino_lib.core.data_augmentations import DataAugmentationDINO
from dino_lib.config.train_config import TrainConfig
from dino_lib.core.dataset import DinoDataset
from dino_lib.core.loss import DINOLoss
from dino_lib.core.functions import train_one_epoch

from python_tools.OSUtils import ensure_dir

IMAGES_FOLDER = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images/')
SPLIT_PATH = os.path.expanduser('~/nanit/dino_data/crop_from_full_resolution_images_data_filter_split.pkl')    # Filtered - Uploading only RGB images
GPU_IDS = 0


def main():
    train_config = TrainConfig()
    ensure_dir(train_config.output_dir)

    # ============ preparing data ... ============
    data_loader = prepare_data(train_config)

    # ============ building student and teacher networks ... ============
    student, teacher, teacher_without_ddp = get_student_teacher_networks(train_config)

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        train_config.out_dim,
        train_config.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        train_config.warmup_teacher_temp,
        train_config.teacher_temp,
        train_config.warmup_teacher_temp_epochs,
        train_config.epochs,
    ).cuda()
    print(f"Loss ready.")

    # ============ preparing optimizer ... ============
    fp16_scaler, optimizer = get_optimizer(student, train_config)

    # ============ init schedulers ... ============
    lr_schedule, momentum_schedule, wd_schedule = init_schedulers(data_loader, train_config)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(train_config.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, train_config.epochs):
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, train_config)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_config': train_config,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(train_config.output_dir, 'checkpoint.pth'))
        if train_config.saveckp_freq and epoch % train_config.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(train_config.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(train_config.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def init_schedulers(data_loader, train_config):
    lr_schedule = utils.cosine_scheduler(
        train_config.lr * (train_config.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        train_config.min_lr,
        train_config.epochs, len(data_loader),
        warmup_epochs=train_config.warmup_epochs,
        start_warmup_value=train_config.min_lr
    )
    wd_schedule = utils.cosine_scheduler(
        train_config.weight_decay,
        train_config.weight_decay_end,
        train_config.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(train_config.momentum_teacher, 1,
                                               train_config.epochs, len(data_loader))
    print(f"Schedulers ready.")
    return lr_schedule, momentum_schedule, wd_schedule


def get_optimizer(student, train_config):
    params_groups = utils.get_params_groups(student)
    if train_config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif train_config.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif train_config.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        raise AttributeError
    # for mixed precision training
    fp16_scaler = None
    if train_config.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    print(f"Optimizer ready.")
    return fp16_scaler, optimizer


def get_student_teacher_networks(train_config):
    student = vits.__dict__[train_config.vit_arch](
        patch_size=train_config.patch_size,
        drop_path_rate=train_config.drop_path_rate,  # stochastic depth
    )
    teacher = vits.__dict__[train_config.vit_arch](patch_size=train_config.patch_size)
    embed_dim = student.embed_dim
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        train_config.out_dim,
        use_bn=train_config.use_bn_in_head,
        norm_last_layer=train_config.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, train_config.out_dim, train_config.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[GPU_IDS])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.DataParallel(student, device_ids=[GPU_IDS])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {train_config.vit_arch} network.")
    return student, teacher, teacher_without_ddp


def prepare_data(train_config):
    transform = DataAugmentationDINO(
        train_config.global_crops_scale,
        train_config.local_crops_scale,
        train_config.local_crops_number,
    )
    dataset = DinoDataset(IMAGES_FOLDER, SPLIT_PATH, 'train', is_train=True, data_aug=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config.batch_size_per_gpu,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")
    return data_loader


if __name__ == '__main__':
    main()
