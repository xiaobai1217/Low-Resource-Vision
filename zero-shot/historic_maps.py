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
today_list = []
today_dict = {}
with open(data_path + 'historic-maps/test.csv') as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        sample_list.append(row)
        if row[0] not in today_list:
            today_dict[row[0]] = len(today_list)
            today_list.append(row[0])
f.close()

today_image_paths = [data_path + "historic-maps/Satellite/" + img_name for img_name in today_list]
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(today_image_paths, device),
}
with torch.no_grad():
    embeddings = model(inputs)
today_embeddings = embeddings[ModalityType.VISION]

old_image_paths = [data_path + "historic-maps/Satellite/" + img_name[1] for img_name in sample_list]
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(old_image_paths, device),
}
with torch.no_grad():
    embeddings = model(inputs)
old_embeddings = embeddings[ModalityType.VISION]

cols = []
for i in range(old_embeddings.size()[0]):
    anchor = old_embeddings[i:i+1]
    similarity = anchor @ today_embeddings.T
    similarity = similarity[0].detach().cpu().numpy()
    sorted_ids = np.argsort(similarity)[::-1]
    gt_id = today_dict[sample_list[i][0]]
    position = np.where(sorted_ids == gt_id)
    cols.append(position)

cols = np.array(cols)
cols = cols.reshape((-1,))

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
print(metrics)