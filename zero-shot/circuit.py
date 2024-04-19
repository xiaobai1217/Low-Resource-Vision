import sys
sys.path.append('/path-to/ImageBind')
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pdb
import csv
import numpy as np

data_path = "/path-to-low-resource/"
class_list = []
class_name_list = []
with open(data_path + "circuits/label_map.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        class_list.append(int(row[0]))
        class_name_list.append(row[1])
f.close()

sample_list = {}
labels = []
with open(data_path + "circuits/test_5_shots.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        sample_list[row[1]] = int(row[2])
        labels.append(int(row[2]))
f.close()
key_list = list(sample_list.keys())

text_inputs = [f"a circuit diagram of {c}" for c in class_name_list]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_inputs, device),
}

with torch.no_grad():
    embeddings = model(inputs)

text_embeddings = embeddings[ModalityType.TEXT]

image_paths = []
acc = 0
for key_name in key_list:
    image_paths.append(data_path + "circuits/data/" + key_name)
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)
image_embeddings = embeddings[ModalityType.VISION]

results = torch.softmax(image_embeddings @ text_embeddings.T, dim=-1)
_, predict = torch.max(results.detach().cpu(), dim=1)
labels = torch.Tensor(np.array(labels))
acc1 = (predict == labels.cpu()).sum().item()
print(acc1 / len(labels))
