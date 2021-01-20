import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score, precision_score, f1_score

import torch

from utils.visualize import visualize_metrics

def evaluate_metric(gt_labels, pred_labels, threshold, epoch, CONFIG, visualize=False, log_info=""):
    """
    Evaluate F1 score, auroc
    """
    metrics = {}
    gt_labels = gt_labels.cpu().detach().numpy()
    pred_labels = pred_labels.cpu().detach().numpy()

    error_index = torch.zeros(len(CONFIG.labels_name), pred_labels.shape[0])

    total_precision = 0
    total_recall = 0

    for i, label_name in enumerate(CONFIG.labels_name):
        gt_label = gt_labels[:, i]
        pred_label = pred_labels[:, i]

        fpr, tpr, _ = roc_curve(gt_label, pred_label)
        roc_auc = auc(fpr, tpr)

        if visualize:
            visualize_metrics(pred_label, gt_label, CONFIG, label_name, epoch, threshold[i], log_info=log_info)

        # F1
        pred_label[pred_label >= threshold[i]] = 1
        pred_label[pred_label < threshold[i]] = 0

        error_index[i, np.logical_and(pred_label==1, gt_label==1)] = 1

        recall = recall_score(gt_label, pred_label, average="macro")
        precision = precision_score(gt_label, pred_label, average="macro")
        f1 = f1_score(gt_label, pred_label, average="macro")

        total_precision += precision
        total_recall += recall

        metrics[label_name] = [roc_auc, recall, precision, f1]

    total_precision /= len(CONFIG.labels_name)
    total_recall /= len(CONFIG.labels_name)
    total_f1 = 2 * (total_recall * total_precision) / (total_recall + total_precision + 1e-8)
    metrics["total"] = [0, total_recall, total_precision, total_f1]

    return total_f1, metrics, error_index
