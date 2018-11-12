# import matlab.engine
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, get_folders_name
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from random import randint
from util.util import Noise, get_dnd_srgb
import os
from scipy.ndimage import gaussian_filter
import cv2
import sys
import matlab.engine
import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()


class T2tDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dataset_mode = True
        self.dir_imnet = os.path.join(self.root, opt.phase + 'A')
        self.imnet_paths = make_dataset(self.dir_imnet)
        self.imnet_paths = sorted(self.imnet_paths)




        transform_ = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_)

        self.crop_size = (256, 256)

    def add_text(self,img):
        import string

        img = np.array(img)

        img = img.copy()
        h, w, _ = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_for_cnt = np.zeros((h, w), np.uint8)
        occupancy = np.random.uniform(0, 35)

        while True:
            n = random.randint(5, 10)
            random_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])
            font_scale = np.random.uniform(0.5, 1)
            thickness = random.randint(1, 3)
            (fw, fh), baseline = cv2.getTextSize(random_str, font, font_scale, thickness)
            x = random.randint(0, max(0, w - 1 - fw))
            y = random.randint(fh, h - 1 - baseline)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.putText(img, random_str, (x, y), font, font_scale, color, thickness)
            cv2.putText(img_for_cnt, random_str, (x, y), font, font_scale, 255, thickness)

            if (img_for_cnt > 0).sum() > h * w * occupancy / 100:
                break

        return Image.fromarray(np.uint8(img))

    def __getitem__(self, index):

        a, b = self.crop_size


        imnet_path_1 = self.imnet_paths[index % len(self.imnet_paths)]

        imnet_path_2 = imnet_path_1
        imnet_path_3 = self.imnet_paths[len(self.imnet_paths) - (index % len(self.imnet_paths)) - 1]

        A1_img = Image.open(imnet_path_1).convert('RGB')
        A1_img = A1_img.resize((a, b))
        B_img = Image.open(imnet_path_3).convert('RGB')
        B_img = B_img.resize((a,b))
        A2_img = Image.open(imnet_path_2).convert('RGB')
        A2_img = A2_img.resize((a, b))

        A1_img = self.add_text(A1_img)
        A2_img = self.add_text(A2_img)

        A1 = self.transform(A1_img)
        A2 = self.transform(A2_img)
        B = self.transform(B_img)




        return {'A1': A1, 'A2': A2, 'B': B, 'A1_paths': imnet_path_1, 'A2_paths': imnet_path_2,'B_paths': imnet_path_3}

    def __len__(self):
        return self.opt.max_dataset_size


    def name(self):
        return 'T2tDataset'

