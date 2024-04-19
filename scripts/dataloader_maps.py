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

class MapsDataset(Dataset):
    def __init__(self, split='train', aug_prob = 0.3, aug_weight = 0.65):
        self.base_path = '/path-to/low-resource/'
        self.data = collections.defaultdict(list)
        with open(self.base_path + 'historic-maps/' + split + '.csv') as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                key = row[0]
                self.data[key].append(row[1])
        f.close()
        self.key_list = list(self.data.keys())

        self.aug_path = '/path-to/stable-diffusion/outputs/historic-maps/' + split + '/strength-0-1/samples'
        self.options = ['7', '75', '8', '85', '9', '95', '99']
        self.contrastive_path = '/path-to/stable-diffusion/outputs/historic-maps/aug/'
        self.aug_prob = aug_prob
        self.aug_weight = aug_weight
        self.split = split

    def __getitem__(self, index):
        key = self.key_list[index]
        random_id = random.randint(0, len(self.data[key])-1)
        old_name = self.data[key][random_id]

        today_im = data.load_and_transform_vision_data([self.base_path + "historic-maps/Satellite/" + key,], "cpu")[0]
        old_im = data.load_and_transform_vision_data([self.base_path + "historic-maps/Satellite/" + old_name,], "cpu")[0]

        weight = 1.0
        old_name_base = old_name.split('.')[0]

        if self.split == 'train' and random.random() < self.aug_prob:
            rand_id = random.randint(0,4)
            old_im = data.load_and_transform_vision_data([os.path.join(self.aug_path, old_name_base, '%d.png'%rand_id),], "cpu")[0]
            weight = self.aug_weight #
        
        if self.split == 'train':
            rand_option = random.randint(0,6)
            rand_aug_id = random.randint(0,4)
            aug_ori_path = os.path.join('/path-to/stable-diffusion/outputs/historic-maps/' + self.split, 'strength-0-'+self.options[rand_option], 'samples', old_name_base, '%d.png'%rand_aug_id)
            aug_ori = data.load_and_transform_vision_data([aug_ori_path,], "cpu")[0]

            rand_aug_aug_id = random.randint(1, 10)
            aug_aug = os.path.join(self.contrastive_path, self.options[rand_option], old_name_base, str(rand_aug_id), '%d.png'%rand_aug_aug_id)
            aug_aug = data.load_and_transform_vision_data([aug_aug,], "cpu")[0]
        else:
            aug_ori = today_im
            aug_aug = today_im

        return today_im, old_im, weight, aug_ori, aug_aug

    def __len__(self):
        return len(self.key_list)

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(MapsDataset(split='train'), batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    
    for i, (today_im, old_im, weight, aug_ori, aug_aug) in enumerate(train_loader):
        pdb.set_trace()