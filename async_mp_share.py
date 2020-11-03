from syft.workers.node_client import NodeClient
import logging
import sys
import asyncio
import torch.nn as nn
import torch.nn.functional as F
import time
import pdb

from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils

from multiprocessing import Process
import argparse
import os
import syft as sy
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms

node_num = int(sys.argv[1])
LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")

# ip mapping
flvm_ip = {
    2: "10.0.17.6",
    3: "10.0.17.17",
    4: "10.0.17.4",
    5: "10.0.17.12",
    6: "10.0.17.14",
    7: "10.0.17.3",
    8: "10.0.17.13",
    9: "10.0.17.8",
    10: "10.0.17.5",
    11: "10.0.17.10",
    12: "10.0.17.28",
    13: "10.0.17.37"
}


"""
class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)
"""
class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, node_num)
        self.fc2 = nn.Linear(node_num, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


async def main():
    hook = sy.TorchHook(torch)
    device = torch.device("cpu")
    optimizer = "SGD"
    epochs = 1
    shuffle = True
    model = Net()
    model.build(torch.zeros([1, 1, 28, 28], dtype=torch.float).to(device))
    # model.build(torch.zeros([2], dtype=torch.float).to(device))

    @sy.func2plan(args_shape=[(-1, 1), (-1, 1)])
    def loss_fn(target, pred):
        return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()

    batch_size = 64
    lr = 0.1
    learning_rate = lr
    optimizer_args = {"lr" : lr}
    model_config = sy.ModelConfig(model=model,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              batch_size=batch_size,
                              optimizer_args=optimizer_args,
                              epochs=epochs,
                              shuffle=shuffle)

    # alice = NodeClient(hook, "ws://172.16.179.20:6666" , id="alice")
    # bob = NodeClient(hook, "ws://172.16.179.21:6667" , id="bob")
    # charlie = NodeClient(hook, "ws://172.16.179.22:6668", id="charlie")
#     testing = NodeClient(hook, "ws://localhost:6669" , id="testing")

    # worker_list = [alice, bob, charlie]

    worker_list = []
    for i in range(2, 2+3):
        worker = NodeClient(hook, "ws://"+flvm_ip[i]+":6666" , id="flvm-"+str(i))
        worker_list.append(worker)

    for worker in worker_list:
        model_config.send(worker)
    grid = sy.PrivateGridNetwork(*worker_list)

    num_of_parameters = len(model.parameters())
    return_ids = []
    for i in range(num_of_parameters):
        return_ids.append("p" + str(i))

    start = time.time()
    # worker_0 = worker_list[0]
    # worker_1 = worker_list[1]
    # worker_2 = worker_list[2]
    enc_results = await asyncio.gather(
        *[
            worker.async_model_share(worker_list, return_ids=return_ids) for worker in worker_list
        ]
    )
    end = time.time()

    ## aggregation
    dst_enc_model = enc_results[0]
    agg_start = time.time()
    with torch.no_grad():
        for i in range(len(dst_enc_model)):
            layer_start = time.time()
            for j in range(1, len(enc_results)):
                add_start = time.time()
                dst_enc_model[i] += enc_results[j][i]
                print("[PROF]", "AddParams", time.time() - add_start)
            print("[PROF]", "Layer"+str(i), time.time() - layer_start)
    print("[PROF]", "AggTime", time.time() - agg_start)




if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(name)s | %(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.INFO)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    asyncio.get_event_loop().run_until_complete(main())
#     main()

