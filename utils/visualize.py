import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from utils.gradcam import GradCam, get_cam_on_image
from utils.util import check_dir_exist

def visualize_metrics(pred_output, gt_output, CONFIG, classes, epoch, threshold, log_info=""):
    pred_output_true = pred_output[gt_output == 1]
    pred_output_false = pred_output[gt_output == 0]

    fig, ax = plt.subplots()
    
    bins = np.linspace(0, 1, 20)
    
    plt.hist(pred_output_true, bins=bins, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none', label="Label : 1")
    plt.hist(pred_output_false, bins=bins, alpha=0.5, histtype='stepfilled', color='red', edgecolor='none', label="Lable : 0")
    plt.axvline(x=threshold, linewidth=4, ls="--")

    check_dir_exist(os.path.join(CONFIG.path_to_save_metrics, str(epoch)))
    
    ax.grid()
    axes = plt.gca()
    plt.legend()
    fig.savefig(os.path.join(CONFIG.path_to_save_metrics, str(epoch), "{}_{}_{}.png".format(classes, log_info, epoch)))



def visualize_gradcam(model, val_loader, visualize_loader, error_indexs, CONFIG, device):
    """
    Get the attention map for the wrong classes
    """
    model.module.set_state(False)
    grad_cam = GradCam(model.module, feature_module=model.module.stages,\
            target_layer_names=["18"], use_cuda=CONFIG.cuda)
    output_images = []

    check_dir_exist(os.path.join(CONFIG.path_to_grid_image))

    for i, label_name in enumerate(CONFIG.labels_name):
        check_dir_exist(os.path.join(CONFIG.path_to_grid_image, label_name))

        for step, (val_datas, visualize_datas) in enumerate(zip(val_loader, visualize_loader)):
            val_X, _, _, _, _ = val_datas
            visualize_X, _, _, _, _ = visualize_datas

            error_index = error_indexs[i, step:step+val_X.shape[0]]
            if torch.sum(error_index) == 0:
                # This batch all correct
                continue
            val_X = val_X[error_index == 1]
            val_X = val_X.to(device)
            
            visualize_X = visualize_X[error_index == 1]       

            for j, (val_X_i, visualize_X_i) in enumerate(zip(val_X, visualize_X)):
                if j == 3:
                    break
                val_X_i = val_X_i.unsqueeze_(0)
                mask = grad_cam(val_X_i, index=1)

                mask = mask[0].cpu().detach()
                mask = mask.permute(1, 2, 0).numpy()
                                                         
                cam_image = get_cam_on_image(visualize_X_i, mask)
                plt.imsave(os.path.join(CONFIG.path_to_grid_image, label_name, "{}_{}_{}.png".format(i, step, j)), cam_image)


