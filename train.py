import os
import argparse
import time
import pandas as pd

import operator
from tqdm import tqdm
from functools import reduce
from scipy import stats

import torch
from torch.utils.data import DataLoader
from tools.DatasetLoadClass import Trainset, Validset
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn, optim


from nets.network import draftNet, ResNet50, ResNet34, ResNet18, ResNet101, ResNet152

from tools.utils import adjust_learning_rate, AverageMeter, save_model_torch, write_log_train, write_log_valid

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main():
    global opt
    cuda = opt.cuda
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print("device:", device)

    seed = 1334
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    print("==========> Loading datasets")

    train_dataset = Trainset(opt.train, opt.label_column, transform=transforms.Compose([
        transforms.RandomCrop(1024, pad_if_needed=True),
        # transforms.Resize(1024),
        transforms.ToTensor(),
    ]))

    valid_dataset = Validset(opt.valid, opt.label_column, transform=transforms.Compose([
        transforms.RandomCrop(1024, pad_if_needed=True),
        # transforms.Resize(1024),
        transforms.ToTensor(),
    ]))

    train_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize,
                                   pin_memory=True, shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_dataset, num_workers=opt.threads, batch_size=opt.batchSize,
                                   pin_memory=True, shuffle=True)

    print("==========> Building model")

    backbone = models.resnet34(pretrained=True)
    model = ResNet34(backbone, num_classes=15)

    criterion = nn.L1Loss()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['state_dict'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("==========> Setting GPU")
    model = model.to(device)
    criterion = criterion.to(device)

    print("==========> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    print("==========> Training")

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    save_name = "{}-{}-({})".format(model.__class__.__name__, opt.label_column, current_time)
    path = os.path.join("./saved-models/", save_name)

    old = -1
    old = valid(valid_data_loader, model, opt.start_epoch - 1, device, save_name)
    max_epoch = opt.start_epoch - 1

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        train(train_data_loader, model, criterion, optimizer, epoch, device, save_name)
        val = valid(valid_data_loader, model, epoch, device, save_name)

        if val >= old:
            filename = "epoch_{}.pt".format(epoch)
            save_model_torch(path, filename, model, optimizer, epoch)
            old = val
            max_epoch = epoch
        elif epoch % 10 == 0:
            filename = "epoch_{}.pt".format(epoch)
            save_model_torch(path, filename, model, optimizer, epoch)
    max_output = "Maximum valid correlation is {}, epoch = {}".format(old, max_epoch)
    print(max_output)
    write_log_valid("./logs/0109", save_name, max_output)


def train(data_loader, model, criterion, optimizer, epoch, device, log_save_name):
    adjust_learning_rate(optimizer, epoch)
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    Logs = AverageMeter()
    t1 = time.time()
    model.train()
    for iteration, (data, label) in enumerate(tqdm(data_loader, desc="Train: "), 1):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        steps = len(data_loader) * (epoch - 1) + iteration

        output = model(data)
        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        Logs.update(loss.cpu())

        torch.cuda.empty_cache()
    t2 = time.time()
    train_log = "(Train) epoch: {}, time: {}, lr: {}, loss: {}".format(
        epoch, t2 - t1, optimizer.param_groups[0]["lr"], Logs.avg)
    print(train_log)
    write_log_train("./logs/0109", log_save_name, train_log)


@torch.no_grad()
def valid(data_loader, model, epoch, device, log_save_name):
    model.eval()
    Cor = AverageMeter()
    t1 = time.time()
    for iteration, (data, label) in enumerate(tqdm(data_loader, desc="Valid: "), 1):
        data = data.to(device)

        output = model(data)
        output = (1 - output) * 15
        output = output.cpu()

        output = output.numpy()
        label = label.numpy()

        label = reduce(operator.add, label)
        output = reduce(operator.add, output)

        output1 = pd.Series(output)
        output2 = output1.rank()
        output3 = output2.values

        value = stats.spearmanr(label, output3)
        cor = value.correlation
        Cor.update(cor)

        # print()
        # print(label)
        # print(output3)
        # print(cor)

        torch.cuda.empty_cache()
    t2 = time.time()
    valid_log = "(Valid) epoch: {}, time: {}, correlation: {}".format(epoch, t2 - t1, Cor.avg)
    print(valid_log)
    write_log_valid("./logs/1224", log_save_name, valid_log)
    return Cor.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Image Process Assignment 1 (pytorch: 1.7.1)")
    parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
    parser.add_argument("--cuda", action="store_true", help="Use cuda?")
    parser.add_argument("--gpus", type=int, default=1, help="nums ofz gpu to use")
    parser.add_argument("--label_column", type=str, help="label column name(Color/Exposure/Noise/Texture)",
                        default='Color')
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning Rate. Default=1e-4")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--step", type=int, default=500, help="step to test the model performance. Default=500")
    parser.add_argument("--tag", type=str, help="tag for this training", default='color')
    parser.add_argument("--train", default="../data/train", type=str,
                        help="path to load train datasets(default: none)")
    parser.add_argument("--test", default="../data/test", type=str,
                        help="path to load test datasets(default: none)")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
    parser.add_argument("--valid", default="../data/valid", type=str,
                        help="path to load valid datasets(default: none)")

    opt = parser.parse_args()
    print(opt)

    os.system('clear')
    main()
