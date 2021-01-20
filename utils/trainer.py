import os
import time
import logging

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score, precision_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import AverageMeter, save
from utils.gradcam import GradCam
from utils.evaluate import evaluate_metric


class Trainer:
    def __init__(self, criterion, optimizer, scheduler, device, CONFIG, *args, **kwargs):
        self.losses = AverageMeter()
        self.seg_losses = AverageMeter()

        self.device = device

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.CONFIG = CONFIG

        self.epochs = self.CONFIG.epochs
        self.threshold = np.array(self.CONFIG.threshold)


    def train_loop(self, train_loader, test_loader, model, fold):
        best_f1 = 0.0
        for epoch in range(self.epochs):
            logging.info("Learning Rate: {:.4f}".format(self.optimizer.param_groups[0]["lr"]))
            logging.info("Start to train for epoch {}".format(epoch))

            self._training_step(model, train_loader, epoch, info_for_logger="_train_step_")

            f1_avg, error_index = self.validate(model, test_loader, epoch)
            if best_f1 < f1_avg:
                logging.info("Best f1 score by now. Save model")
                best_f1 = f1_avg
                save(model, self.optimizer, self.CONFIG.path_to_save_model[:-4]+"_{}".format(fold)+self.CONFIG.path_to_save_model[-4:])

        logging.info("The Best f1 score : {}".format(best_f1))


    def _training_step(self, model, loader, epoch, info_for_logger=""):
        model.train()
        start_time = time.time()

        balance_sample = self.CONFIG.balance_sample
        model.module.set_state(self.CONFIG.seg_state)

        gt_labels = []
        pred_labels = []

        for step, (X, y, seg_labels) in enumerate(loader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            self.optimizer.zero_grad()

            model.module.set_state(self.CONFIG.seg_state)
            self.optimizer.zero_grad()

            seg_loss = None
            if self.CONFIG.seg_state:
                seg_labels = seg_labels.to(self.device, non_blocking=True)
                outs, first_seg_out, second_seg_out, third_seg_out = model(X)
                seg_loss = self.criterion.segmentation_loss(first_seg_out, second_seg_out, third_seg_out, seg_labels)
            else:
                outs = model(X)

            loss = self.criterion(outs, y, balance_sample)
            loss = loss + seg_loss if self.CONFIG.seg_state else loss
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            gt_labels.append(y)
            pred_labels.append(F.sigmoid(outs))

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train", seg_loss=seg_loss)

        # =========== Evaluate =========================
        gt_labels = torch.cat(gt_labels)
        pred_labels = torch.cat(pred_labels)

        total_f1, metrics, _ = evaluate_metric(gt_labels, pred_labels, self.threshold, epoch, self.CONFIG, visualize=True, log_info="train")
        # ==============================================

        self._epoch_stats_logging(metrics, start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train="Train")
        for avg in [self.losses]:
            avg.reset()


    def validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()
        model.module.set_state(False)

        gt_labels = []
        pred_labels = []

        with torch.no_grad():
            for step, (X, y, _) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]

                outs = model(X)

                loss = self.criterion(outs, y)
                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")

                gt_labels.append(y)
                pred_labels.append(F.sigmoid(outs))

        gt_labels = torch.cat(gt_labels)
        pred_labels = torch.cat(pred_labels)

        # =========== Evaluate =========================
        total_f1, metrics, error_index = evaluate_metric(gt_labels, pred_labels, self.threshold, epoch, self.CONFIG, visualize=True, log_info="val")
        # ==============================================

        self._epoch_stats_logging(metrics, start_time=start_time, epoch=epoch, val_or_train="val")
        for avg in [self.losses]:
            avg.reset()

        return total_f1, error_index

    def _epoch_stats_logging(self, metrics, start_time, epoch, val_or_train, info_for_logger=""):
        for k, v in metrics.items():
            auc, recall, precision, f1 = v
            logging.info(info_for_logger+val_or_train+":[{:3d}/{}] Label Name {}, AUC {:.4f}, Recall {:.4f}, Precision {:.4f}, F1 {:.4f}, Time {:.2f}"\
                    .format(epoch+1, self.epochs, k, auc, recall, precision, f1, time.time()-start_time))


    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train, seg_loss=None):
        self.losses.update(loss.item(), N)
        if seg_loss is not None:
            self.seg_losses.update(seg_loss.item(), N)

        if (step > 1 and step % self.CONFIG.print_freq==0) or step == len_loader -1 :
            logging.info(val_or_train+
                    ":[{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} Seg Loss {:.3f}".format(
                        epoch+1, self.epochs, step, len_loader-1, self.losses.get_avg(), self.seg_losses.get_avg(),
                        ))

