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
from torch.hub import load_state_dict_from_url
import pdb
import resnet
import syft as sy
hook = sy.TorchHook(torch)

def select_samples(original_dataset, keep_labels):

    ## transform to torch.tensor'
    original_dataset.data = torch.from_numpy(original_dataset.data)

    marks = np.isin(original_dataset.targets, keep_labels).astype("uint8")

    indices = [i for i,x in enumerate(marks) if x == 1]

    selected_data = torch.stack(list(original_dataset.data[i] for i in indices) ,dim =0)
    selected_targets = [original_dataset.targets[i] for i in indices]

    ## transform back to numpy
#     selected_data = selected_data.numpy()

    return (selected_data, selected_targets)

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

my_resnet18 = resnet.resnet18()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

selected_data, selected_targets = select_samples(cifar10_dataset, keep_labels)
dataset = sy.BaseDataset(data=selected_data, targets=selected_targets, transform=cifar10_dataset.transform)

net = my_resnet18
trainloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                          shuffle=True, num_workers=2)
train_loss = 0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05,
                      momentum=0.9, weight_decay=5e-4)
kbar = pkbar.Kbar(target=len(trainloader), epoch=0, num_epochs=1, width=8, always_stateful=False)

pdb.set_trace()
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
#     print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
