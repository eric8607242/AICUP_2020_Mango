import os 
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
from utils.mango_dataset import MangoDataset

def get_transforms(CONFIG):
    train_transform = Compose([
            Resize(CONFIG.input_size, CONFIG.input_size),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
    val_transform = Compose([
            Resize(CONFIG.input_size, CONFIG.input_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    test_transform = Compose([
            Resize(CONFIG.input_size, CONFIG.input_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    return train_transform, val_transform, test_transform


def get_mango(fold_train_data, fold_val_data, train_transform, val_transform, test_transform, CONFIG):
    defect_classes = CONFIG.defect_classes if CONFIG.defect_classification else []

    train_data = MangoDataset(train_data_info, train_root_path, CONFIG.labels_name, transforms=train_transform, defect_classes=defect_classes)
    val_data = MangoDataset(val_data_info, val_root_path, CONFIG.labels_name, transforms=val_transform)
    test_data = MangoDataset(test_data_info, test_root_path, CONFIG.labels_name, transforms=test_transform)

    return train_data, val_data, test_data


def get_dataset(train_transform, val_transform, test_transform, CONFIG):
    train_dataset, val_dataset, test_dataset = get_mango(train_transform, val_transform, test_transform, CONFIG)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(train_dataset, val_dataset, test_dataset, CONFIG):
    def _build_loader(dataset, shuffle):
        return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=CONFIG.batch_size,
                    pin_memory=True,
                    num_workers=CONFIG.num_workers,
                    shuffle=shuffle,
                )

    train_loader = _build_loader(train_dataset, True)
    val_loader = _build_loader(val_dataset, True)
    test_loader = _build_loader(test_dataset, False)

    return train_loader, val_loader, test_loader


def merge_train_dev_data(dataset_dir):
    train_root_path = os.path.join(dataset_dir, "Train")
    dev_root_path = os.path.join(dataset_dir, "Dev")

    train_csv_path = os.path.join(dataset_dir, "train.csv")
    dev_csv_path = os.path.join(dataset_dir, "dev.csv")

    train_data_info = pd.read_csv(train_csv_path, header=None)
    dev_data_info = pd.read_csv(dev_csv_path, header=None)

    def _add_root_path(x, root_path):
        return os.path.join(root_path, x)

    train_data_info[0] = train_data_info[0].apply(_add_root_path, root_path=train_root_path)
    dev_data_info[0] = dev_data_info[0].apply(_add_root_path, root_path=dev_root_path)

    train_all_data_info = pd.concat((train_data_info, dev_data_info), ignore_index=True)
    return train_all_data_info
