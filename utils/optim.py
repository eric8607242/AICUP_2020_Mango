import logging 

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_lr_scheduler(optimizer, step_per_epoch, CONFIG):
    logging.info("================ Scheduler =================")
    logging.info("Scheduler : {}".format(CONFIG.lr_scheduler))

    if CONFIG.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_per_epoch*CONFIG.epochs)

    elif CONFIG.lr_scheduler == "step":
        logging.info("Step size : {}".format(CONFIG.step_size))
        logging.info("Gamma : {}".format(CONFIG.decay_ratio))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG.step_size*step_per_epoch, gamma=CONFIG.decay_ratio, last_epoch=-1)

    return lr_scheduler


def get_optimizer(model_parameters, CONFIG, log_info=""):
    logging.info("================= Optimizer =================")
    logging.info("{} Optimizer : {}".format(log_info, CONFIG.optimizer))
    logging.info("{} Learning rate : {}".format(log_info, CONFIG.lr))
    logging.info("{} Weight decay : {}".format(log_info, CONFIG.weight_decay))

    if CONFIG.optimizer == "sgd":
        logging.info("{} Momentum : {}".format(log_info, CONFIG.momentum))
        optimizer = torch.optim.SGD(params=model_parameters,
                                    lr=CONFIG.lr,
                                    momentum=CONFIG.momentum,
                                    weight_decay=CONFIG.weight_decay)

    elif CONFIG.optimizer == "rmsprop":
        logging.info("{} Momentum : {}".format(log_info, CONFIG.momentum))
        optimizer = torch.optim.RMSprop(model_parameters,
                            lr=CONFIG.lr,
                            alpha=CONFIG.alpha,
                            momentum=CONFIG.momentum,
                            weight_decay=CONFIG.weight_decay)

    elif CONFIG.optimizer == "adam":
        logging.info("{} Beta : {}".format(log_info, CONFIG.beta))
        optimizer = torch.optim.Adam(model_parameters,
                            weight_decay=CONFIG.weight_decay,
                            lr=CONFIG.lr,
                            betas=(CONFIG.beta, 0.999))

    return optimizer


class Loss:
    def __init__(self, device, CONFIG):
        self.CONFIG = CONFIG
        self.device = device

        self.loss_bce = nn.BCEWithLogitsLoss()

    def __call__(self, outs, y, balance_sample=False):
        """
        Calculate loss for each classes.
        With "balance_classes" flag, the output and label will be reconcatenated to balanced data.
        """
        if balance_sample:
            loss = []
            for i, label_name in enumerate(self.CONFIG.labels_name):
                label_outs = outs[:, i]
                label_y = y[:, i]

                if label_name in self.CONFIG.balance_classes:
                    label_loss = self._balance_loss(label_outs, label_y)
                    if label_loss is None:
                        continue

                    loss.append(label_loss)
                else:
                    label_loss = self.loss_bce(label_outs, label_y)
                    loss.append(label_loss)

            loss = sum(loss)/len(self.CONFIG.labels_name)
        else:
            loss = self.loss_bce(outs, y)

        return loss

    def _balance_loss(self, label_outs, label_y):
        p_label_outs = label_outs[label_y == 1]
        p_label_y = label_y[label_y == 1]
        p_len = p_label_outs.shape[0]

        if p_len == 0:
            # This batch no label_data
            return None

        n_label_outs = label_outs[label_y == 0]
        n_label_y = label_y[label_y == 0]
        n_len = n_label_outs.shape[0]

        label_ratio = min(int(n_len / p_len), 20)
        balance_outs = [p_label_outs for i in range(label_ratio)]
        balance_y = [p_label_y for i in range(label_ratio)]

        balance_outs.append(n_label_outs)
        balance_y.append(n_label_y)

        # Balance sample
        balance_outs = torch.cat(balance_outs)
        balance_y = torch.cat(balance_y)

        # Calculate loss
        label_loss = self.loss_bce(balance_outs, balance_y)
        return label_loss

    def segmentation_loss(self, first_seg_out, second_seg_out, third_seg_out, y):
        first_y = F.interpolate(y, size=[64, 64])
        second_y = F.interpolate(y, size=[32, 32])
        third_y = F.interpolate(y, size=[16, 16])

        first_loss = self.loss_bce(first_seg_out, first_y) * self.CONFIG.seg_loss_weight
        second_loss = self.loss_bce(second_seg_out, second_y) * self.CONFIG.seg_loss_weight
        third_loss = self.loss_bce(third_seg_out, third_y) * self.CONFIG.seg_loss_weight

        return (first_loss+ second_loss + third_loss) / 3




