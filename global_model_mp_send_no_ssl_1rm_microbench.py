from syft.workers.node_client import NodeClient
import logging
import sys
import asyncio
import torch.nn as nn
import torch.nn.functional as F
import time
import pdb
import sys

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
import gevent
from syft.frameworks.torch.fl.loss_fn import nll_loss

LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")
node_num = int(sys.argv[1])

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

"""
class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(node_num, 1)
        self.fc2 = nn.Linear(1, node_num)

    def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
"""

def send_model_to_worker(
    worker,
    built_model: sy.Plan,
):
    """Send the model to the worker and fit the model on the worker's training data.

    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        curr_round: Index of the current training round (for logging purposes).
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        lr: Learning rate of each training step.

    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    built_model.id = "GlobalModel"

    model_send_start = time.time()
    # pdb.set_trace()
    built_model.send(worker)
    print("[trace] GlobalModelSend duration", worker.id, time.time() - model_send_start)

    return None


def main():
    hook = sy.TorchHook(torch)
    device = torch.device("cpu")
    model = Net()
    model.build(torch.zeros([1, 1, 28, 28], dtype=torch.float).to(device))
    # model.build(torch.zeros([1, node_num], dtype=torch.float).to(device))

    @sy.func2plan()
    def loss_fn(pred, target):
        return nll_loss(input=pred, target=target)

    input_num = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    dummy_pred = F.log_softmax(input_num, dim=1)
    loss_fn.build(dummy_pred, target)

    built_model = model
    built_loss_fn = loss_fn

    epoch_num = 21
    batch_size = 64
    lr = 0.1
    learning_rate = lr
    optimizer_args = {"lr" : lr}

    alice = NodeClient(hook, "ws://172.16.179.20:6666" , id="alice")
    # bob = NodeClient(hook, "ws://172.16.179.21:6667" , id="bob")
    # charlie = NodeClient(hook, "ws://172.16.179.22:6668", id="charlie")

    worker_list = [alice]
    # worker_list = [alice]
    grid = sy.PrivateGridNetwork(*worker_list)

    for epoch in range(epoch_num):

        logger.info("round %s/%s", epoch, epoch_num)

        epoch_start = time.time()

        jobs = [
            gevent.spawn(
                send_model_to_worker,
                worker,
                built_model
            )for worker in worker_list
        ]

        gevent.joinall(jobs)

        # results = await asyncio.gather(
        #     *[
        #         send_model_to_worker(
        #             worker=worker,
        #             built_model=built_model,
        #         )
        #         for worker in worker_list
        #     ]
        # )
        print("[PROF]", "AllWorkerSend", "duration", "COORD", time.time() - epoch_start)

        built_model.pointers = {}
        built_loss_fn.pointers = {}

if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

#     asyncio.get_event_loop().run_until_complete(main())
    main()

