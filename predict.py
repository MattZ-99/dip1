import argparse
import time
import pandas as pd

import operator
from tqdm import tqdm
from functools import reduce
from scipy import stats

import torch
from torch.utils.data import DataLoader
from tools.DatasetLoadClass import Testset, Validset
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn, optim

from nets.network import draftNet, ResNet50, ResNet34, ResNet18, ResNet101, ResNet152

from tools.utils import adjust_learning_rate, AverageMeter, save_model_torch, write_log_train, write_log_valid

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def output_predict_result(output, name, dir="./output_test/"):
    values = []
    for i in range(len(output[0])):
        v = []
        for j in range(len(output)):
            v.append(output[j][i])
        values.append(v)
    columns = ["Color", "Exposure", "Noise", "Texture"]
    alphabet_name = ['A_', 'B_', 'C_', 'D_', 'E_', 'F_', 'G_', 'H_', 'I_', 'J_', 'K_', 'L_', 'M_', 'N_', 'O_']
    index = [al + name + '.jpg ' for al in alphabet_name]

    a = pd.DataFrame(values, index=index, columns=columns)
    a.to_csv("output_test2/" + name + '.csv')


cuda = True
if cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
print("device:", device)

seed = 1334
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

print("==========> Loading datasets")

test_dataset = Validset(label_name="Exposure", data_dir="../data/valid", transform=transforms.Compose([
    transforms.RandomCrop(1024, pad_if_needed=True),
    # transforms.Resize(1024),
    transforms.ToTensor(),
]))

test_data_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=1,
                              pin_memory=True, shuffle=True)

print("==========> Loading model")

backbone_color = models.resnet18(pretrained=True)
model_color = ResNet18(backbone_color, num_classes=15)

checkpoint_color = torch.load("saved-models/ResNet18-Color/epoch_72.pt")
model_color.load_state_dict(checkpoint_color["net"])

backbone_exposure = models.resnet34(pretrained=True)
model_exposure = ResNet34(backbone_exposure, num_classes=15)

checkpoint_exposure = torch.load("saved-models/ResNet34-Exposure/epoch_76.pt")
model_exposure.load_state_dict(checkpoint_exposure["net"])

backbone_noise = models.resnet34(pretrained=True)
model_noise = ResNet34(backbone_noise, num_classes=15)

checkpoint_noise = torch.load("saved-models/ResNet34-Noise/epoch_10.pt")
model_noise.load_state_dict(checkpoint_noise["net"])

backbone_texture = models.resnet34(pretrained=True)
model_texture = ResNet34(backbone_texture, num_classes=15)

checkpoint_texture = torch.load("saved-models/ResNet34-Texture/epoch_30.pt")
model_texture.load_state_dict(checkpoint_texture["net"])

model_color.eval()
model_exposure.eval()
model_noise.eval()
model_texture.eval()

model_color = model_color.to(device)
model_exposure = model_exposure.to(device)
model_noise = model_noise.to(device)
model_texture = model_texture.to(device)

models = [model_color, model_exposure, model_noise, model_texture]

print("==========> Predict Start")
avg = []
with torch.no_grad():
    for iteration, (data, name) in enumerate(tqdm(test_data_loader, desc="Test: ")):
        # print(iteration, data.shape, name)
        # name = name[0].split('/')[-1]
        data = data.to(device)
        Out = []
        num = 0

        for model in models:
            num += 1
            output = model(data)
            output = (1 - output) * 15
            output = output.cpu()

            output = output.numpy()

            output = reduce(operator.add, output)

            output1 = pd.Series(output)
            output2 = output1.rank()
            output3 = output2.values

            label = reduce(operator.add, name)
            if num == 2:
                value = stats.spearmanr(label, output3)
                value = value.correlation
                # print(iteration, value)
                avg.append(value)
            # print(output3)
            Out.append(output3)
        # output_predict_result(Out, name)

print(avg, reduce(operator.add, avg)/20)