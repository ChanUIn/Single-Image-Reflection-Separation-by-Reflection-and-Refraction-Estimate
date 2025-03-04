# data_loader.py
from torchvision import transforms as T
import os
from PIL import Image
from torch.utils import data
import random
import torch


def get_loader_t(config):
    transform = []
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform1 = T.Compose(transform)
    transform2 = T.Compose([T.ToTensor()])

    dataset_t = Data_Read(config.main_dir, transform1, transform2, 'train')
    loader_t = data.DataLoader(dataset=dataset_t,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

    return loader_t

def get_loader_r(config):
    transform = []
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform1 = T.Compose(transform)
    transform2 = T.Compose([T.ToTensor()])

    dataset_r = Data_Read_r(config.main_dir, transform1, transform2, 'train')
    loader_r = data.DataLoader(dataset=dataset_r,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

    return loader_r


def get_loader_val(config):
    transform = []
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform1 = T.Compose(transform)
    transform2 = T.Compose([T.ToTensor()])

    dataset = Data_Read(config.main_dir, transform1, transform2, 'val')
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    return data_loader


def get_loader_test(config):
    transform = []
    transform.append(T.Resize((128, 128)))
    transform.append(T.ToTensor())

    transform1 = T.Compose(transform)
    transform2 = T.Compose([T.ToTensor()])

    if config.mode == 'test':
        config.batch_size = 1

    dataset = Data_Read(config.main_dir, transform1, transform2, 'test/18')

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    return data_loader


class Data_Read(data.Dataset):
    def __init__(self, path_main, transform1, transform2, mode):

        self.path_mixture = os.path.join(path_main, mode, 'syn')
        self.path_ground = os.path.join(path_main, mode, 't')
        self.alpha_path = os.path.join(path_main, mode, 'estimate.txt') #
        self.transform1 = transform1
        self.transform2 = transform2
        files = os.listdir(self.path_ground)
        self.num_images = len(files)
        self.files = files
        lines = [line.rstrip() for line in open(self.alpha_path, 'r')]
        lines = lines[0:]
        self.datapair = []
        for i, line in enumerate(sorted(lines)):
            split = line.split()
            filename = split[0]
            alpha = float(split[1])
            self.datapair.append([filename, alpha])
        self.num_images = len(self.datapair)

    def __getitem__(self, index):
        filename, alpha = self.datapair[index]
        fname = self.files[index]
        img_gt = Image.open(os.path.join(self.path_ground, filename)).convert('RGB')
        img_mix = Image.open(os.path.join(self.path_mixture, filename)).convert('RGB')
        if alpha > 1:
            alpha = 1

        return self.transform1(img_mix), self.transform1(img_gt), alpha

    def __len__(self):
        return self.num_images


class Data_Read_r(data.Dataset):
    def __init__(self, path_main, transform1, transform2, mode):

        self.path_mixture = os.path.join(path_main, mode, 'syn')
        self.path_ground = os.path.join(path_main, mode, 'r')
        self.alpha_path = os.path.join(path_main, mode, 'AB.txt')
        self.transform1 = transform1
        self.transform2 = transform2
        files = os.listdir(self.path_ground)
        self.num_images = len(files)
        self.files = files
        lines = [line.rstrip() for line in open(self.alpha_path, 'r')]
        lines = lines[0:]
        self.datapair = []
        for i, line in enumerate(sorted(lines)):
            split = line.split()
            filename = split[0]
            alpha = float(split[1])
            self.datapair.append([filename, alpha])
        self.num_images = len(self.datapair)

    def __getitem__(self, index):
        filename, alpha = self.datapair[index]
        fname = self.files[index]
        img_gt = Image.open(os.path.join(self.path_ground, filename)).convert('RGB')
        img_mix = Image.open(os.path.join(self.path_mixture, filename)).convert('RGB')
        if alpha > 1:
            alpha = 1

        return self.transform1(img_mix), self.transform1(img_gt), alpha

    def __len__(self):
        return self.num_images



"""

def get_loader_train(config):
    transform = []
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform1 = T.Compose(transform)
    transform2 = T.Compose([T.ToTensor()])

    dataset_t = Data_Read(config.main_dir, transform1, transform2, 'train')
    data_loader_t = data.DataLoader(dataset=dataset_t,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

    dataset_r = Data_Read_r(config.main_dir, transform1, transform2, 'train')
    data_loader_r = data.DataLoader(dataset=dataset_r,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

    return data_loader_t, data_loader_r
"""