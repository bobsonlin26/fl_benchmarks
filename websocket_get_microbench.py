import syft as sy
import torch
from syft.workers.node_client import NodeClient
import torch.nn.functional as F
import torch.nn as nn
import time
import sys

node_num = int(sys.argv[1])

# Model
class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(node_num, 1)
        self.fc2 = nn.Linear(1, node_num)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

hook = sy.TorchHook(torch)
alice = NodeClient(hook, "ws://10.0.17.6:6666" , id="flvm-2")


for i in range(21):
    model = Net()
    model.build(torch.zeros([1, node_num], dtype=torch.float))
    ptr_model = model.send(alice)
    start_time = time.time()
    m = ptr_model.get()
    end_time = time.time()

    print("[PROF]", "GetTime", "duration", "COORD", end_time - start_time)

