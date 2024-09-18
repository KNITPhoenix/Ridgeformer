from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision import transforms, utils
from torchvision.utils import save_image
import cv2
import random
# Ignore warnings
import warnings
import os
import json
from PIL import Image
warnings.filterwarnings("ignore")

def cropping_preprocess(image):
    non_zero_pixels = np.where(image != 255)
    y_min, x_min = np.min(non_zero_pixels[0]), np.min(non_zero_pixels[1])
    y_max, x_max = np.max(non_zero_pixels[0]), np.max(non_zero_pixels[1])
    top_left = (x_min, y_min)
    top_right = (x_max, y_min)
    bottom_left = (x_min, y_max)
    bottom_right = (x_max, y_max)
    height = bottom_right[1] - top_left[1] + 1
    width = bottom_right[0] - top_left[0] + 1
    cropped_img = image[top_left[1]:top_left[1] + height, top_left[0]:top_left[0] + width]

    h,w = cropped_img.shape[:2]
    if h>224 and w>224:
        return cropped_img
    else:
        scale_factor_h = 224 / h
        scale_factor_w = 224 / w
        new_width = int(w * scale_factor_w)
        new_height = int(h * scale_factor_h)
        resized_image = cv2.resize(cropped_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # print(resized_image.shape)
        return resized_image

class RandomRegionBlackOut(object):
    def __init__(self, p=0.5, blackout_ratio=0.2):
        self.p = p  # Probability of applying the transform
        self.blackout_ratio = blackout_ratio  # Ratio of the image area to blackout

    def __call__(self, img):
        if random.random() < self.p:
            channels, width, height = img.shape
            mask_width              = int(width * self.blackout_ratio)
            mask_height             = int(height * self.blackout_ratio)

            start_x                 = random.randint(0, width - mask_width)
            start_y                 = random.randint(0, height - mask_height)

            img[:, start_x:start_x+mask_width, start_y:start_y+mask_height] = 0.0

        return img
    
class RandomRegionBlurOut(object):
    def __init__(self, p=0.5, blackout_ratio=0.2):
        self.p = p  # Probability of applying the transform
        self.blackout_ratio = blackout_ratio  # Ratio of the image area to blackout

    def __call__(self, img):
        if random.random() < self.p:
            channels, width, height = img.shape
            mask_width              = int(width * self.blackout_ratio)
            mask_height             = int(height * self.blackout_ratio)

            start_x                 = random.randint(0, width - mask_width)
            start_y                 = random.randint(0, height - mask_height)
            
            img[:, start_x:start_x+mask_width, start_y:start_y+mask_height] = transforms.GaussianBlur((3,3), sigma=(0.1, 2.0))(img[:, start_x:start_x+mask_width, start_y:start_y+mask_height])
        return img

class hktest(Dataset):
    def __init__(self, split = "train"):
        self.split = split
        self.transforms ={
        "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                RandomRegionBlackOut(p=0.4, blackout_ratio=0.2),
                RandomRegionBlurOut(p=0.4, blackout_ratio=0.2),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2)),
                transforms.Grayscale(num_output_channels=3),
                ]),
        "test": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Grayscale(num_output_channels=3),
                ])
        }

        contactless_paths = list()
        contactbased_paths = list()
        contactless_ids = list()
        contactbased_ids = list()
        x=0

        with open("hkpolyu_test.json",'r') as js:
            sample_dict = json.load(js)
        for file in sample_dict:
            contactless_paths.extend(sample_dict[file]['Contactless'])
            contactbased_paths.extend(sample_dict[file]['Contactbased'])
            contactless_ids.append(file)
            contactbased_ids.append(file)

        self.train_files = {
            "contactless": contactless_paths,
            "contactbased": contactbased_paths
        }
        self.transform = self.transforms[split]
        self.allfiles = self.train_files
        self.all_files_paths_contactless = contactless_paths
        self.label_id_mapping = contactless_ids
        self.all_labels = list()
        self.label_id_to_contactbased = dict()
        
        for filename in self.allfiles["contactless"]:
            id = filename.split("/")[-3]
            self.all_labels.append(self.label_id_mapping.index(id))

        for filename in self.allfiles["contactbased"]:
            id = filename.split("/")[-3]
            id = self.label_id_mapping.index(id)
            if (id in self.label_id_to_contactbased):
                self.label_id_to_contactbased[id].append(filename)
            else:
                self.label_id_to_contactbased[id] = [filename]

        print("Number of Contactbased Files: ", len(self.allfiles["contactbased"]))
        print("Number of Contactless Files: ",  len(self.allfiles["contactless"]))
        print("Number of classes: ", len(self.label_id_mapping))
        print("Total number of images ", split ," : ", len(self.all_labels))
        
    def __len__(self):
        return len(self.all_files_paths_contactless)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.all_labels[idx]
        contactless_filename = self.all_files_paths_contactless[idx]
        if len(self.label_id_to_contactbased[label]) == 1:
            contactbased_filename = self.label_id_to_contactbased[label][0]
        else:
            contactbased_filename = self.label_id_to_contactbased[label][idx % len(self.label_id_to_contactbased[label])]
        
        contactless_sample = Image.open(contactless_filename)
        contactless_sample = contactless_sample.convert("RGB")
        contactbased_sample = cv2.imread(contactbased_filename)
        contactbased_sample = cropping_preprocess(contactbased_sample)
        
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                        transforms.Grayscale(num_output_channels=3),
                        ])
        contactless_sample  = self.transform(contactless_sample)
        contactbased_sample = self.transform(contactbased_sample)
        return contactless_sample, contactbased_sample, self.all_labels[idx]
