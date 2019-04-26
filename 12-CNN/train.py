#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:32:05 2019

@author: santiago
"""
# %%
from google_drive_downloader import GoogleDriveDownloader as gdd
import pandas
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn
import tqdm


# %%
def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader),
                                               desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output, target)
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()

    print("Loss: %0.3f | Acc: %0.2f" % (np.array(loss_cum).mean(), float(Acc * 100) / len(data_loader.dataset)))


# %%

class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.maxpool = nn.AdaptiveMaxPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        self.Loss = nn.BCEWithLogitsLoss()


# %%


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))


# %%

if __name__ == '__main__':

    if os.path.isdir('12-CNN/celeba-dataset') == False:
        gdd.download_file_from_google_drive(file_id='1ybgNi6RGDKjem13N66ur1umXfR1RAiDH',
                                            dest_path='12-CNN/celeba-dataset.zip',
                                            unzip=True)
    # %%
    info = pandas.read_csv('celeba-dataset/list_attr_celeba.csv')
    imgs = os.listdir('celeba-dataset/img_align_celeba')
    part = pandas.read_csv('celeba-dataset/list_eval_partition.csv')

    train = []
    val = []
    test = []
    train_lab = []
    val_lab = []
    for idx in range(len(imgs)):
        idd = part.partition[idx]
        img = part.image_id[idx]
        if idd == 0:
            train.append(img)
        if idd == 1:
            test.append(img)
        if idd == 2:
            val.append(img)
    ##############################################################################
    # %%
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    image_size = 227
    batch_size = 4 * 8
    workers = 2
    ngpu = 1

    dataset = dset.ImageFolder(root='celeba-dataset',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    # %%
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    plot = False
    if plot:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Examples")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig('dataset.png')
    # %%
    model = Net()
    model.to(device)

    model.training_params()
    print_network(model, 'Conv network')

    # Exploring model
    data, _ = next(iter(dataloader))
    _ = model(data.to(device).requires_grad_(False))
    #%%

   for epoch in range(5):
       train(train_loader, model, epoch)
       if TEST: test(test_loader, model, epoch)
