import numpy as np
import torch as th
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
import syft as sy
from syft.frameworks.torch.fl.loss_fn import nll_loss
import time
import sys

hook = sy.TorchHook(th)

device = th.device("cpu")
worker_id = sys.argv[1]
batch_size = int(sys.argv[2])
lr = 0.1
epoch_num = 10

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

class Net(sy.Plan):
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
model.build(th.zeros([1, 1, 28, 28], dtype=th.float).to(device))

@sy.func2plan()
def loss_fn(pred, target):
    return nll_loss(input=pred, target=target)

input_num = th.randn(3, 5, requires_grad=True)
target = th.tensor([1, 0, 4])
dummy_pred = F.log_softmax(input_num, dim=1)
loss_fn.build(dummy_pred, target)

optimizer = getattr(th.optim, "SGD")
optimizer_args = {"lr" : lr}
optimizer_args.setdefault("params", model.parameters())
optimizer = optimizer(**optimizer_args)


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

print("[PROF]", "NR_DATA", "number", worker_id, len(dataset))

## plan_fit
data_range = range(len(dataset))
sampler = RandomSampler(data_range)
data_loader = th.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )

for _ in range(epoch_num):

    start_time = time.time()
    fetch_start = time.time()
    for (data, target) in data_loader:

        fetch_end = time.time()
        print("[PROF]", "FetchBatch", "duration", worker_id, fetch_end - fetch_start)

        optimizer.zero_grad()
        output = model(data.to(device))
        loss = loss_fn(output, target.to(device))
        loss.backward()
        optimizer.step()

        fetch_start = time.time()

        print("[PROF]", "CompBatch", "duration", worker_id, fetch_start - fetch_end)

    end_time = time.time()
    print("[PROF]", "LocalTraining", "duration", worker_id, end_time - start_time)
