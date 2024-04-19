import sys
sys.path.append('/path-to/ImageBind')
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pdb
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression


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
train_labels = []
with open(data_path + "circuits/train_5_shots.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        sample_list[row[1]] = int(row[2])
        train_labels.append(int(row[2]))
f.close()
key_list = list(sample_list.keys())

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

image_paths = []
for key_name in key_list:
    image_paths.append(data_path + "circuits/data/" + key_name)
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)
train_features = embeddings[ModalityType.VISION]


sample_list = {}
test_labels = []
with open(data_path + "circuits/test_5_shots.csv") as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        sample_list[row[1]] = int(row[2])
        test_labels.append(int(row[2]))
f.close()
key_list = list(sample_list.keys())

image_paths = []
for key_name in key_list:
    image_paths.append(data_path + "circuits/data/" + key_name)
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)
test_features = embeddings[ModalityType.VISION]

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
train_features = train_features.detach().cpu().numpy()
test_features = test_features.detach().cpu().numpy()

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")