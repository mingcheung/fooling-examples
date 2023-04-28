import os
import csv
import torch
import torch.nn as nn
from imageio import imsave


# load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


# save ndarray to images.
def save_images(images, filenames, save_dir):
    for i, filename in enumerate(filenames):
        print("Saving file {}...".format(filename))
        save_path = os.path.join(save_dir, filename+".png")
        imsave(save_path, images[i, :, :, :], format="png")


# import numpy as np
#
# image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')
# filenames, _, _ = load_ground_truth("dataset/images.csv")
# X_adv = np.load("exp_results/ensemble-based-attacks/X_adv_Res50_Inc-v3_Dense121_VGG16_300iters_Logit_gaussian.npy")
# X_adv_img = np.transpose(X_adv, (0, 2, 3, 1))
# save_images(X_adv_img, filenames, "dataset/adv_images/X_adv_Res50_Inc-v3_Dense121_VGG16_300iters_Logit_gaussian")
