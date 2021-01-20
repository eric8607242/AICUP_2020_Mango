import os

import numpy as np
import pandas as pd
from skimage import io

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

class MangoDataset(Dataset):
    def __init__(self, data_info, root_dir, labels_name, transforms=None, defect_classes=[]):
        self.data_info = data_info
        self.root_dir = root_dir

        self.transforms = transforms
        self.labels_name = labels_name
    
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item_info = self.data_info.iloc[idx]

        image_name = item_info[0]
        image_path = os.path.join(self.root_dir, image_name)

        label_vector, positions = self._process_labels(item_info[1:])

        image = Image.open(image_path)
        image = image.convert('RGB')

        attention_label = self._get_attention_label(positions, image)
        attention_label = attention_label.numpy()

        if self.transforms:
            image = np.array(image)
            augmented = self.transforms(image=image, mask=attention_label)
            image, attention_label = augmented["image"], augmented["mask"]
            attention_label = attention_label.permute(2, 0, 1)

        return image, label_vector, attention_label

    def _get_attention_label(self, positions, image):
        max_w, max_h = image.size
        attention_label = torch.zeros((max_h, max_w, 1), dtype=torch.float32)

        for p_row in range(positions.shape[0]):
            x, y, w, h, label_index = positions[p_row]
            x, y, w, h, label_index = \
                    x.astype(np.int), y.astype(np.int), w.astype(np.int), h.astype(np.int), label_index.astype(np.int)
            if label_index == -1:
                continue

            start_y = int(max(0, y-max_h*0.1))
            start_x = int(max(0, x-max_w*0.1))

            bounder_h = int(min(max_h, y+h+max_h*0.1))
            bounder_w = int(min(max_w, x+w+max_w*0.1))
            attention_label[start_y:bounder_h, start_x:bounder_w, :] = 1

        return attention_label

    def _process_labels(self, image_labels):
        labels_len = len(image_labels) // 5

        label_vector = np.zeros(5) # labels
        position = np.zeros((labels_len, 5)) # x, y, w, h, label_index
        position += -1

        for i in range(labels_len):
            x, y, w, h, label = image_labels[i*5:(i+1)*5]
            if type(label) != str:
                break

            label_index = self.labels_name.index(label)
            label_vector[label_index] = 1

            position[i] = x, y, w, h, label_index

        return label_vector.astype(np.float32), position
            


class TestMangoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.data_info = pd.read_csv(csv_file, header=None)
        self.data_info = self.data_info.iloc[1:]
        self.root_dir = root_dir

        self.transforms = transforms

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item_info = self.data_info.iloc[idx]

        image_name, x, y, w, h = item_info
        image_path = os.path.join(self.root_dir, image_name)

        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.transforms:
            image = np.array(image)
            image = self.transforms(image=image)

        return image

            


