import os
import time
import logging
import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils.config import get_config
from utils.util import get_logger, set_random_seed, load_state_dict
from utils.dataflow import get_transforms, get_dataloader
from utils.mango_dataset import MangoDataset
from utils.optim import get_optimizer, get_lr_scheduler, Loss
from utils.trainer import Trainer
from utils.model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the config file", required=True)
    parser.add_argument("--load-pretrained", type=bool, help="load pretrained weight", default=True)
    parser.add_argument("--title", type=str, help="experiment title", required=True)
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    if CONFIG.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    set_random_seed(CONFIG.seed)
    get_logger(CONFIG.log_dir)

    logging.info("=================================== Experiment title : {} Start ===========================".format(args.title))

    #train_root_path = os.path.join(CONFIG.dataset_dir, "Train_all")
    #train_csv_path = os.path.join(CONFIG.dataset_dir, "train_all.csv")
    #train_data_info = pd.read_csv(train_csv_path, header=None)
    train_data_info = merge_train_dev_data(CONFIG.dataset_dir)

    folds = StratifiedKFold(n_splits=5, shuffle=True).split(np.arange(train_data_info.shape[0]), train_data_info.iloc[:, 5])

    train_transform, val_transform, test_transform = get_transforms(CONFIG)
    defect_classes = CONFIG.defect_classes if CONFIG.defect_classification else []

    for fold, (trn_idx, val_idx) in enumerate(folds):
        logging.info("Fold : {}".format(fold))
        
        fold_train_data = train_data_info.iloc[trn_idx]
        fold_val_data = train_data_info.iloc[val_idx]

        #train_data = MangoDataset(fold_train_data, train_root_path, CONFIG.labels_name, transforms=train_transform, defect_classes=defect_classes)
        train_data = MangoDataset(fold_train_data, CONFIG.dataset_dir, CONFIG.labels_name, transforms=train_transform, defect_classes=defect_classes)
        val_data = MangoDataset(fold_val_data, CONFIG.dataset_dir, CONFIG.labels_name, transforms=val_transform)

        train_loader, val_loader, test_loader = get_dataloader(train_data, val_data, val_data, CONFIG)

        model = Model(input_size=CONFIG.input_size, classes=CONFIG.classes, se=True, activation="hswish", l_cfgs_name=CONFIG.model, seg_state=CONFIG.seg_state)

        if args.load_pretrained:
            pretrained_dict = load_state_dict(CONFIG.model_pretrained, use_ema=CONFIG.ema)
            model.load_state_dict(pretrained_dict, strict=False)
            logging.info("Load pretrained from {} to {}".format(CONFIG.model_pretrained, CONFIG.model))

        if (device.type == "cuda" and CONFIG.ngpu >= 1):
            model = model.to(device)
            model = nn.DataParallel(model, list(range(CONFIG.ngpu)))

        optimizer = get_optimizer(model.parameters(), CONFIG.optim_state)
        criterion = Loss(device, CONFIG)
        scheduler = get_lr_scheduler(optimizer, len(train_loader), CONFIG)

        start_time = time.time()
        trainer = Trainer(criterion, optimizer, scheduler, device, CONFIG)
        trainer.train_loop(train_loader, test_loader, model, fold)

    logging.info("Total training time : {:.2f}".format(time.time() - start_time))
    logging.info("=================================== Experiment title : {} End ===========================".format(args.title))
        

