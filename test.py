import os
import time
import logging
import argparse

import pandas as pd

import torch
import torch.nn as nn

from utils.config import get_config
from utils.util import get_logger, set_random_seed, load_state_dict
from utils.dataflow import get_transforms, get_dataset, get_dataloader
from utils.mango_dataset import TestMangoDataset
from utils.optim import get_optimizer, get_lr_scheduler, Loss
from utils.trainer import Trainer
from utils.model import Model
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", type=int, default=1)
    parser.add_argument("--cfg", type=str, help="path to the config file", required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    if CONFIG.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    test_transform = Compose([
            Resize(CONFIG.input_size, CONFIG.input_size),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    test_root_path = os.path.join(CONFIG.dataset_dir, "Test")
    test_csv_path = os.path.join(CONFIG.dataset_dir, "test.csv")
    test_dataset = TestMangoDataset(test_csv_path, test_root_path, transforms=test_transform)
    test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=CONFIG.batch_size,
                        pin_memory=True,
                        num_workers=CONFIG.num_workers,
                        shuffle=False
                    )

    set_random_seed(CONFIG.seed)

    tta_pred_labels = []
    for i, path in enumerate([CONFIG.path_to_save_model[:-4]+"_{}".format(i)+CONFIG.path_to_save_model[-4:] for i in range(5)]):
        model = Model(input_size=CONFIG.input_size, classes=CONFIG.classes, se=True, activation="hswish", l_cfgs_name=CONFIG.model, seg_state=CONFIG.seg_state)
        pretrained_dict = load_state_dict(path)
        model.load_state_dict(pretrained_dict["model"], strict=False)

        if (device.type == "cuda" and CONFIG.ngpu >= 1):
            model = model.to(device)
            model = nn.DataParallel(model, list(range(CONFIG.ngpu)))
        model.module.set_state(False)

        with torch.no_grad():
            for t in range(args.tta):
                pred_labels = []
                model.eval()
                for step, X in enumerate(test_loader):
                    X = X["image"]
                    X = X.to(device, non_blocking=True)

                    outs = model(X)
                    pred_labels.append(F.sigmoid(outs))
                pred_labels = torch.cat(pred_labels)
                tta_pred_labels.append(pred_labels)

    tta_pred_labels = sum(tta_pred_labels) / (args.tta*6)
    tta_pred_labels = tta_pred_labels.cpu().numpy()
    tta_pred_labels[tta_pred_labels > 0.5] = 1
    tta_pred_labels[tta_pred_labels <= 0.5] = 0
    tta_pred_labels = tta_pred_labels.astype(int)

    upload_sheet = pd.read_csv("Test_UploadSheet.csv")
    upload_sheet.iloc[:, 1:] = tta_pred_labels
    upload_sheet.to_csv(args.output_file, index=False)

