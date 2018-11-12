import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, get_folders_name
from PIL import Image
import torchvision.transforms as transforms
from random import randint
import os





class R2nDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dataset_mode = True
        self.dir_A = os.path.join(self.root, opt.phase + 'A')
        self.dir_B = os.path.join(self.root, opt.phase + 'B')



        self.A_folders_names = get_folders_name(self.dir_A)
        self.A_folders_names = sorted(self.A_folders_names)

        self.B_paths = make_dataset(self.dir_B)
        self.B_paths = sorted(self.B_paths)

        self.B_size = len(self.B_paths)

        transform_ = [transforms.ToTensor(),
                      transforms.Normalize([0.5, 0.5, 0.5],
                                           [0.5, 0.5, 0.5])]

        transform_2 = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_)
        self.transform2 = transforms.Compose(transform_2)
        self.crop_size = (256, 256)

    def sliding_window(self,im, stepSize=50):
        windowSize = self.crop_size
        w, h = im.size
        for y in range(0, h-windowSize[1], stepSize):
            for x in range(0, w-windowSize[0], stepSize):

                yield x,y

    def __getitem__(self, index):

        a, b = self.crop_size

        A_folder = self.A_folders_names[index % self.B_size]
        A_paths = make_dataset(A_folder)
        A2_paths = A_paths

        i = randint(0, len(A_paths) - 1)
        j = randint(0, len(A_paths) - 1)

        A1_path = A_paths[i]
        A2_path = A2_paths[j]

        B_path = self.B_paths[index % self.B_size]


        A1_img = Image.open(A1_path).convert('RGB')
        A2_img = Image.open(A2_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        l = list(self.sliding_window(A1_img))
        x, y = l[index % len(l)]

        l = list(self.sliding_window(A2_img))
        x1, y1 = l[index % len(l)]

        A1_img = A1_img.crop((x, y, x + a, y + b))
        A2_img = A2_img.crop((x1, y1, x1 + a, y1 + b))
        B_img = B_img.crop((x, y, x + a, y + b))








        A1 = self.transform(A1_img)
        A2 = self.transform(A2_img)
        B = self.transform(B_img)





        return {'A1': A1, 'A2': A2, 'B': B,'A1_paths': A1_path, 'A2_paths': A2_path,'B_paths': B_path}

    def __len__(self):
        return self.opt.max_dataset_size


    def name(self):
        return 'R2nDataset'








