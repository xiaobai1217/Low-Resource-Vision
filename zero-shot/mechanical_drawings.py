import os
import sys
sys.path.append('/path-to/ImageBind')
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pdb
import csv
import numpy as np
import scipy

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

data_path = "/path-to/low-resource/"

sample_list = []
with open(data_path + 'mechanical-drawing/test.csv') as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        sample_list.append(row)
f.close()

image_paths = [data_path + "mechanical-drawing/data/" + img_name[0] + '-a.png' for img_name in sample_list]
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
}
with torch.no_grad():
    embeddings = model(inputs)
a_embeddings = embeddings[ModalityType.VISION]

image_paths = [data_path + "mechanical-drawing/data/" + img_name[0] + '-b.png' for img_name in sample_list]
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
}
with torch.no_grad():
    embeddings = model(inputs)
b_embeddings = embeddings[ModalityType.VISION]

cols = []
for i in range(len(sample_list)):
    anchor = a_embeddings[i:i+1]
    similarity = anchor @ b_embeddings.T 
    similarity = similarity[0].detach().cpu().numpy()
    sorted_ids = np.argsort(similarity)[::-1]
    position = np.where(sorted_ids == i)
    cols.append(position)
cols = np.array(cols)
cols = cols.reshape((-1,))

cols2 = []
for i in range(len(sample_list)):
    anchor = b_embeddings[i:i+1]
    similarity = anchor @ a_embeddings.T 
    similarity = similarity[0].detach().cpu().numpy()
    sorted_ids = np.argsort(similarity)[::-1]
    position = np.where(sorted_ids == i)
    cols2.append(position)
cols2 = np.array(cols2)
cols2 = cols2.reshape((-1,))


def cols2metrics(cols, num_queries):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1 
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics

metrics = cols2metrics(cols, len(sample_list))
metrics2 = cols2metrics(cols2, len(sample_list))

print('R1:', (metrics['R1'] + metrics2['R1'])/2.0)
print('R5:', (metrics['R5'] + metrics2['R5'])/2.0)
print('R10:', (metrics['R10'] + metrics2['R10'])/2.0)
print('MedR:', (metrics['MedR'] + metrics2['MedR'])/2.0)
print('MeanR:', (metrics['MeanR'] + metrics2['MeanR'])/2.0)