import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import sys

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pkbar
import pdb

device = torch.device("cpu")
worker_id = "alice"
batch_size = 64
lr = 0.1
epoch_num = 20

KEEP_LABELS_DICT = {
 "alice": [0, 1, 2, 3],
 "bob": [4, 5, 6],
 "charlie": [7, 8, 9],
 "testing": list(range(10)),
 None: list(range(10)),
}

keep_labels = KEEP_LABELS_DICT[worker_id]
resnet18 = models.resnet18()
net = resnet18

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=64,
                                          shuffle=True, num_workers=1)

net.train()
train_loss = 0
correct = 0
total = 0

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05,
                      momentum=0.9, weight_decay=5e-4)

kbar = pkbar.Kbar(target=len(trainloader), epoch=0, num_epochs=1, width=8, always_stateful=False)

for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    kbar.update(batch_idx, values=[("loss", train_loss/(batch_idx+1)), ("Acc", 100.*correct/total)])
