import numpy as np
import torch as th
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
import syft as sy
from syft.frameworks.torch.fl.loss_fn import nll_loss
import torch.optim as optim
import time
import sys

hook = sy.TorchHook(th)

device = th.device("cpu")
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

key = "mnist"
training = True

if training == True:
    key = "mnist"
else:
    key = "mnist_testing"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()

keep_labels = KEEP_LABELS_DICT[worker_id]
mnist_dataset = datasets.MNIST(
     root="./data",
     train=training,
     download=True,
     transform=transforms.Compose(
         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
     ),
 )
indices = np.isin(mnist_dataset.targets, keep_labels).astype("uint8")
selected_data = (
     th.native_masked_select(mnist_dataset.data.transpose(0, 2), th.tensor(indices))
     .view(28, 28, -1)
     .transpose(2, 0)
 )
selected_targets = th.native_masked_select(mnist_dataset.targets, th.tensor(indices))
dataset = sy.BaseDataset(
 data=selected_data, targets=selected_targets, transform=mnist_dataset.transform)

dataset = sy.BaseDataset(data=selected_data, targets=selected_targets, transform=mnist_dataset.transform)

trainloader = th.utils.data.DataLoader(dataset, batch_size=64,
                                          shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=0.001)

start_time = time.time()
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()
    optimizer.step()
print("[PROF]", "LocalTraining", "duration", time.time() - start_time)
