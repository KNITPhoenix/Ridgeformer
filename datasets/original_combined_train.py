from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import InterpolationMode
import cv2
import random
# Ignore warnings
import warnings
import os
import json
warnings.filterwarnings("ignore")
from torchvision.utils import save_image
from PIL import Image

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

class Combined_original(Dataset):
    def __init__(self, manifest_files, split = "train"):
        self.base_path = "manifest_jsons/"
        self.manifest_file = manifest_files
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
        for files_path in self.manifest_file:
            with open(os.path.join(self.base_path,files_path),'r') as js:
                sample_dict = json.load(js)
            for file in sample_dict:
                contactless_paths.extend(sample_dict[file]['Contactless'])
                contactbased_paths.extend(sample_dict[file]['Contactbased'])
                contactless_ids.append(file+"_"+os.path.splitext(files_path)[0])
                contactbased_ids.append(file+"_"+os.path.splitext(files_path)[0])

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
        self.all_datasets_id = list()
        
        for filename in self.allfiles["contactless"]:
            if filename.split("/")[-4].lower() == 'ridgebase':
                id = filename.split("/")[-3]+"_"+filename.split("/")[-2].split("_")[2].lower()+"_"+filename.split("/")[-2].split("_")[4].lower()+"_"+filename.split("/")[-4].lower()
                self.all_datasets_id.append(0)
            elif filename.split("/")[-5] in ['ISPFDv2_colorback','ISPFDv2_blackback']:
                id = filename.split("/")[-3]+"_"+filename.split("/")[-2]+"_"+filename.split("/")[-5].lower()
                self.all_datasets_id.append(1)
            elif filename.split("/")[-4].lower() == 'hkpolyu':
                id = filename.split("/")[-3]+"_"+filename.split("/")[-4].lower()
                self.all_datasets_id.append(2)
            else:
                id = filename.split("/")[-3]+"_"+filename.split("/")[-2]+"_"+filename.split("/")[-4].lower()
                self.all_datasets_id.append(3)
            self.all_labels.append(self.label_id_mapping.index(id))

        for filename in self.allfiles["contactbased"]:
            if filename.split("/")[-4].lower() == 'ridgebase':
                id = filename.split("/")[-3]+"_"+filename.split("/")[-2].split("_")[2].lower()+"_"+filename.split("/")[-2].split("_")[4].lower()+"_"+filename.split("/")[-4].lower()
            elif filename.split("/")[-5] in ['ISPFDv2_colorback','ISPFDv2_blackback']:
                id = filename.split("/")[-3]+"_"+filename.split("/")[-2]+"_"+filename.split("/")[-5].lower()
            elif filename.split("/")[-4].lower() == 'hkpolyu':
                id = filename.split("/")[-3]+"_"+filename.split("/")[-4].lower()
            else:
                id = filename.split("/")[-3]+"_"+filename.split("/")[-2]+"_"+filename.split("/")[-4].lower()
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
        
        # reading image
        if contactless_filename.split("/")[-4].lower() == 'hkpolyu':
            contactless_sample = Image.open(contactless_filename)
            contactless_sample = contactless_sample.convert("RGB")
            contactbased_sample = cv2.imread(contactbased_filename)
            contactbased_sample = cropping_preprocess(contactbased_sample)
        else:
            contactless_sample = cv2.imread(contactless_filename)
            contactbased_sample = cv2.imread(contactbased_filename)

        # flipping
        if contactless_filename.split("/")[-4].lower() == 'ridgebase':
            hand  = contactless_filename.split("/")[-2].split("_")[2].lower()
            if hand == "right":
                contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_CLOCKWISE)
                contactless_sample = cv2.flip(contactless_sample, 1)
            else:
                contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_COUNTERCLOCKWISE)
                contactless_sample = cv2.flip(contactless_sample, 1)
        elif contactless_filename.split("/")[-4].lower() in ['hkpolyu'] and self.split != 'train':
            contactless_sample = cv2.flip(contactless_sample, 1)
        elif contactless_filename.split("/")[-4] == 'ISPFDv1_blackback' or contactless_filename.split("/")[-4] == 'ISPFDv1_colorback': # for the consistency of combined data
            contactless_sample = cv2.rotate(contactless_sample, cv2.ROTATE_90_CLOCKWISE)
            contactbased_sample = cv2.rotate(contactbased_sample, cv2.ROTATE_90_CLOCKWISE)
        
        # transformations for image
        if contactless_filename.split("/")[-4].lower() == 'ridgebase':   # done
            category_cb = 0
            category_cl = 1
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                    RandomRegionBlackOut(p=0.4, blackout_ratio=0.2),
                    RandomRegionBlurOut(p=0.2, blackout_ratio=0.1),
                    transforms.ColorJitter(brightness=0.6, contrast=2),
                    transforms.Grayscale(num_output_channels=3)
                    ])
            contactless_sample  = self.transform(contactless_sample)
            contactbased_sample = self.transform(contactbased_sample)
        elif contactless_filename.split("/")[-5] in ['ISPFDv2_blackback']:   # done
            category_cb = 2
            category_cl = 3
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                    RandomRegionBlackOut(p=0.4, blackout_ratio=0.2),
                    RandomRegionBlurOut(p=0.2, blackout_ratio=0.1),
                    transforms.Grayscale(num_output_channels=3)
                    ])
            contactless_sample  = self.transform(contactless_sample)
            contactless_sample  = transforms.functional.adjust_brightness(contactless_sample,brightness_factor = 0.1)
            contactless_sample  = transforms.functional.autocontrast(contactless_sample)
            contactbased_sample = self.transform(contactbased_sample)
        elif contactless_filename.split("/")[-4].lower() == 'hkpolyu':
            category_cb = 4
            category_cl = 5
            self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                        RandomRegionBlackOut(p=0.4, blackout_ratio=0.2),
                        transforms.RandomRotation((-10,10)),
                        transforms.Grayscale(num_output_channels=3),
                        ])
            contactless_sample  = self.transform(contactless_sample)
            contactless_sample  = transforms.functional.adjust_brightness(contactless_sample,brightness_factor = 0.4)
            contactless_sample  = transforms.functional.autocontrast(contactless_sample)
            contactbased_sample = self.transform(contactbased_sample)
        elif contactless_filename.split("/")[-5] in ['ISPFDv2_colorback']:   # done
            category_cb = 2
            category_cl = 3
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                    RandomRegionBlackOut(p=0.4, blackout_ratio=0.2),
                    RandomRegionBlurOut(p=0.2, blackout_ratio=0.1),
                    transforms.Grayscale(num_output_channels=3)
                    ])
            contactless_sample  = self.transform(contactless_sample)
            contactless_sample  = transforms.functional.adjust_brightness(contactless_sample,brightness_factor = 0.4)
            contactless_sample  = transforms.functional.autocontrast(contactless_sample)
            contactbased_sample = self.transform(contactbased_sample)
        elif contactless_filename.split("/")[-4] == 'ISPFDv1_blackback':  # done
            category_cb = 6
            category_cl = 7
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                    RandomRegionBlackOut(p=0.4, blackout_ratio=0.2),
                    RandomRegionBlurOut(p=0.2, blackout_ratio=0.1),
                    transforms.ColorJitter(brightness=0.8, contrast=3),
                    transforms.Grayscale(num_output_channels=3)
                    ])
            contactless_sample  = self.transform(contactless_sample)
            contactbased_sample = self.transform(contactbased_sample)
        else:   # ISPFDv1_colorback - done
            category_cb = 6
            category_cl = 7
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                    RandomRegionBlackOut(p=0.4, blackout_ratio=0.2),
                    RandomRegionBlurOut(p=0.2, blackout_ratio=0.1),
                    transforms.Grayscale(num_output_channels=3)
                    ])
            contactless_sample  = self.transform(contactless_sample)
            contactless_sample  = transforms.functional.adjust_brightness(contactless_sample,brightness_factor = 0.6)
            contactless_sample  = transforms.functional.adjust_contrast(contactless_sample,contrast_factor = 2)
            contactbased_sample = self.transform(contactbased_sample)

        return contactless_sample, contactbased_sample, self.all_labels[idx], category_cl, category_cb
