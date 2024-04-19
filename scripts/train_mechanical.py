import pdb
from dataloader_mechanical import MechanicalDrawingDataset
import torch
import argparse
import tqdm
import os
import numpy as np
import math
import csv
import collections
import torch.nn as nn
import random
import warnings
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F
import sys
sys.path.append('path-to-directory/demo-code/')
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def train_one_step(a_imgs, b_imgs, weights, aug_imgs, aug_imgs2, feature_bank):
    a_features, _ = model(a_imgs)
    b_features, _ = model(b_imgs)
    a_features = F.normalize(a_features, dim=-1)
    b_features = F.normalize(b_features, dim=-1)

    b_contrastive_logits = args.factor * b_features @ a_features.t()
    a_contrastive_logits = args.factor * a_features @ b_features.t()
    
    contrastive_labels = torch.arange(a_imgs.size()[0], dtype=torch.long)
    contrastive_labels = contrastive_labels.cuda(non_blocking=True)
    loss = (torch.sum(F.cross_entropy(a_contrastive_logits, contrastive_labels, reduce=False) * weights) / torch.sum(weights) + torch.sum(F.cross_entropy(b_contrastive_logits, contrastive_labels, reduce=False) * weights) / torch.sum(weights)) / 2

    aug_features, _ = model(aug_imgs)
    aug_features = F.normalize(aug_features, dim=-1)
    aug_features2, _ = model(aug_imgs2)
    aug_features2 = F.normalize(aug_features2, dim=-1)

    if len(feature_bank) > 0:
        compared_features = torch.cat(feature_bank, dim=0)
        compared_features = torch.cat((aug_features, compared_features), dim=0)
        contrastive_logits = args.aug_factor * aug_features2 @ compared_features.t()
    else:
        contrastive_logits = args.aug_factor * aug_features2 @ aug_features.t()

    contrastive_labels = torch.arange(aug_imgs.size()[0], device='cuda', dtype=torch.long)
    contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_labels)
    # pdb.set_trace()
    loss += contrastive_loss * args.lambda

    optim.zero_grad()
    optim.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    feature_bank.append(aug_features.detach())
    if len(feature_bank) > args.bank_size:
        feature_bank = feature_bank[-args.bank_size:]

    return loss, feature_bank

def val_one_step(a_imgs, b_imgs):
    with torch.no_grad():
        a_features, _ = model(a_imgs)
        b_features, _ = model(b_imgs)
        a_features = F.normalize(a_features, dim=-1)
        b_features = F.normalize(b_features, dim=-1)
        b_contrastive_logits = b_features @ a_features.t()
        a_contrastive_logits = a_features @ b_features.t()
        contrastive_labels = torch.arange(a_imgs.size()[0], device='cuda', dtype=torch.long)
        loss = (F.cross_entropy(a_contrastive_logits, contrastive_labels) + F.cross_entropy(b_contrastive_logits, contrastive_labels)) / 2

    return loss

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='input a str', default=5e-4)
    parser.add_argument('--save_id', type=int, help='input a str', default=1)
    parser.add_argument('--resume', type=bool, help='input a str', default=False)
    parser.add_argument('--resume_checkpoint', type=str, help='input a str', default='')
    parser.add_argument('--batch_size', type=int, help='input a str', default=8)
    parser.add_argument('--alpha', type=float, help='input a str', default=0.03)
    parser.add_argument('--bank_size', type=int, help='input a str', default=35)
    parser.add_argument('--aug_prob', type=float, help='input a str', default=0.3)
    parser.add_argument('--aug_weight', type=float, help='input a str', default=0.65)
    parser.add_argument('--aug_factor', type=float, help='input a str', default=80.0)
    parser.add_argument('--factor', type=float, help='input a str', default=80.0)
    parser.add_argument('--lambda', type=float, help='input a str', default=0.1)
    parser.add_argument('--middle_dimension', type=int, help='input a str', default=2)
    parser.add_argument('--epochs', type=int, help='input a str', default=90)
    parser.add_argument('--att_num', type=int, help='input a str', default=10)
    parser.add_argument('--att_layers', type=str, help='input a str', default='16')
    parser.add_argument('--alpha', type=float, help='input a str', default=0.5)
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    warnings.filterwarnings("ignore")

    # assign the desired device.
    device = 'cuda' # or 'cpu'
    device = torch.device(device)

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    log_path = base_path + "log_%02d.csv"%args.save_id
    cmd = ['rm -rf ', log_path]
    os.system(' '.join(cmd))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    att_layers = args.att_layers.split(',')
    att_layers = [int(num) for num in att_layers]
    model = imagebind_model.imagebind_huge(pretrained=True, task_type='retrieval', middle_dimension=args.middle_dimension, att_num = args.att_num, att_layers = att_layers, alpha=args.alpha)    
    model.to(device)
    model = nn.DataParallel(model)
    get_logits = GetLogits()
    get_logits.to(device)
    get_logits = nn.DataParallel(get_logits)

    loss_fn = nn.CrossEntropyLoss()
    trainable_parameters = []
    trainable_parameters += [{'params': model.module.W_downs}]
    trainable_parameters += [{'params': model.module.W_ups}]
    trainable_parameters += [{'params': model.module.adapt_att}]
    trainable_parameters += [{'params': model.module.adapt_vector}]
    trainable_parameters += [{'params': model.module.adapt_alpha}]
    trainable_parameters += [{'params': model.module.adapt_att_tokenization}]

    optim = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=5e-7, betas=(0.95,0.999))

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(MechanicalDrawingDataset(split='train', aug_prob = args.aug_prob, aug_weight = args.aug_weight),batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(MechanicalDrawingDataset(split='val', ),batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    BestLoss = float("inf")
    BestEpoch = 0
    BestAcc = 0
    scaler = GradScaler()
    feature_bank = []
    with open(log_path, "a") as f:
        for epoch_i in range(args.epochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                count = 0
                total_loss = 0
                loss=0
                print(split)
                model.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for i, (a_imgs, b_imgs, weights, aug_imgs, aug_imgs2) in enumerate(dataloaders[split]):
                        a_imgs = a_imgs.type(torch.FloatTensor).cuda()
                        b_imgs = b_imgs.type(torch.FloatTensor).cuda()
                        weights = weights.cuda()
                        aug_imgs = aug_imgs.type(torch.FloatTensor).cuda()
                        aug_imgs2 = aug_imgs2.type(torch.FloatTensor).cuda()

                        if split=='train':
                            loss, feature_bank = train_one_step(a_imgs, b_imgs, weights, aug_imgs, aug_imgs2, feature_bank, alpha=args.alpha)
                        else:
                            loss = val_one_step(a_imgs, b_imgs)

                        total_loss += loss.item() * batch_size

                        count += a_imgs.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}".format(
                                total_loss / float(count),
                                loss.item()))
                        pbar.update()
                    f.write("{},{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), BestEpoch,BestLoss))
                    f.flush()

            if total_loss / float(count) < BestLoss:
                BestLoss = total_loss / float(count)
                BestEpoch = epoch_i
                save = {
                    'epoch': epoch_i,
                    'state_dict': model.state_dict(),
                    'best_loss': BestLoss,
                }

                torch.save(save,
                           base_path + "best%02d.pt"%args.save_id)

        f.write("BestEpoch,{},BestLoss,{}\n".format(BestEpoch, BestLoss))
        f.flush()

    f.close()
