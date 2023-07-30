import torch 
from torch.utils.data import Dataset

import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def plot_heatmap(denorm_image, pred, heatmap):

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20,20), ncols=3)

    classes = ['cucumber', 'eggplant', 'mushroom']
    ps = torch.nn.Softmax(dim = 1)(pred).cpu().detach().numpy()
    ax1.imshow(denorm_image)

    ax2.barh(classes, ps[0])
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    ax3.imshow(denorm_image)
    ax3.imshow(heatmap, cmap='magma', alpha=0.7)


class ImageDataset(Dataset):

    def __init__(self, df, data_dir = None, augs = None,):
        self.df = df
        self.augs = augs
        self.data_dir = data_dir 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = self.data_dir + row.img_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = row.label 

        if self.augs:
            data = self.augs(image = img)
            img = data['image']

        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, label
