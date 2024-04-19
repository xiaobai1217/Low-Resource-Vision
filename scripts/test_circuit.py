import sys
sys.path.append('/home/yzhang8/vision-foundation-models/ImageBind-AdaptFormer-att')
from imagebind import data
import torch
import torch.nn as nn
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pdb
import csv
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_id', type=int, help='input a str', default=1)
    parser.add_argument('--middle_dimension', type=int, help='input a str', default=2)
    parser.add_argument('--att_num', type=int, help='input a str', default=10)
    parser.add_argument('--att_layers', type=str, help='input a str', default='12')
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_path = "/home/yzhang8/technical-dataset/"
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    att_layers = args.att_layers.split(',')
    att_layers = [int(num) for num in att_layers]
    model = imagebind_model.imagebind_huge(pretrained=True, middle_dimension=args.middle_dimension, att_num = args.att_num, att_layers = att_layers)
    model.to(device)
    model = nn.DataParallel(model)
    checkpoint = torch.load("checkpoints/best%02d.pt"%args.save_id)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    image_paths = []
    acc = 0
    for key_name in key_list:
        image_paths.append(data_path + "circuits/data/" + key_name)
    images = data.load_and_transform_vision_data(image_paths, device)

    outputs = []
    batch_size = 10
    with torch.no_grad():
        for i in range(len(images) // batch_size + 1):
            embeddings, _ = model(images[i*batch_size:(i+1)*batch_size])
            outputs.append(embeddings)
    results = torch.cat(outputs, dim=0)#[ModalityType.VISION]

    #results = torch.softmax(image_embeddings @ text_embeddings.T, dim=-1)
    _, predict = torch.max(results.detach().cpu(), dim=1)
    labels = torch.Tensor(np.array(labels))
    acc1 = (predict == labels.cpu()).sum().item()
    print(acc1 / len(labels))
#pdb.set_trace()

# label_count = np.zeros((32, ))
# label_count2 = np.zeros((32, ))
# labels = labels.cpu().numpy()
# predict = predict.cpu().numpy()
# for i in range(32):
#     if predict[i] == labels[i]:
#         label_count[int(labels[i])] += 1
#     label_count2[int(labels[i])] += 1
# for i in range(32):
#     print(i, label_count[i] / label_count2[i])
#     # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#     # values, indices = similarity[0].topk(5)

#     # gt = sample_list[key_name]
#     # if values[0].item() == gt:
#     #     acc += 1


# pdb.set_trace()