from torch.utils.data import Dataset
import torch
import random
import numpy as np
import csv
import torchvision.transforms as transforms
from PIL import Image
import pdb
import collections
import sys
sys.path.append('/path-to-directory/demo-code/')
from imagebind import data
import os

class MechanicalDrawingDataset(Dataset):
    def __init__(self, split='train', aug_prob = 0.3, aug_weight = 0.65):
        self.base_path = '/path-to/low-resource/'
        self.data = []
        with open(self.base_path + 'mechanical-drawing/' + split + '.csv') as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                self.data.append(row)
        f.close()

        self.aug_path = '/path-to/stable-diffusion/outputs/mechanical-drawing/' + split + '/strength-0-1/samples'
        self.options = ['7', '75', '8', '85', '9', '95', '99']
        self.contrastive_path = '/path-to/stable-diffusion/outputs/mechanical-drawing/aug/'
        self.aug_prob = aug_prob
        self.aug_weight = aug_weight
        self.split = split

    def __getitem__(self, index):
        id = self.data[index][0]
        a_im = data.load_and_transform_vision_data([self.base_path + "mechanical-drawing/data/" + id + '-a.png',], "cpu")[0]
        b_im = data.load_and_transform_vision_data([self.base_path + "mechanical-drawing/data/" + id + '-b.png',], "cpu")[0]

        weight = 1.0

        if self.split == 'train' and random.random() < self.aug_prob:
            rand_id = random.randint(0,4)
            if random.random() < 0.5:
                rand_ab = 'a'
                a_im = data.load_and_transform_vision_data([os.path.join(self.aug_path, str(id) + '-' + rand_ab, '%d'%rand_id + '.png'),], "cpu")[0]
            else:
                rand_ab = 'b'
                b_im = data.load_and_transform_vision_data([os.path.join(self.aug_path, str(id) + '-' + rand_ab, '%d'%rand_id + '.png'),], "cpu")[0]
            weight = self.aug_weight #
        
        if self.split == 'train':
            rand_option = random.randint(0,6)
            rand_aug_id = random.randint(0,4)
            aug_ori_path = os.path.join('/path-to/stable-diffusion/outputs/mechanical-drawing/' + self.split, 'strength-0-'+self.options[rand_option], 'samples', str(id) + '-a', '%d.png'%rand_aug_id)
            aug_ori = data.load_and_transform_vision_data([aug_ori_path,], "cpu")[0]

            rand_aug_aug_id = random.randint(1, 10)
            aug_aug = os.path.join(self.contrastive_path, self.options[rand_option], str(id) + '-a', str(rand_aug_id), '%d.png'%rand_aug_aug_id)
            aug_aug = data.load_and_transform_vision_data([aug_aug,], "cpu")[0]
        else:
            aug_ori = a_im
            aug_aug = a_im

        return a_im, b_im, weight, aug_ori, aug_aug

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(CadDataset(split='train'), batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    
    for i, (a_im, b_im, weight, aug_ori, aug_aug) in enumerate(train_loader):
        pdb.set_trace()