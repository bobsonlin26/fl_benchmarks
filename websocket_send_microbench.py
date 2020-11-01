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

from syft.frameworks.torch.fl.loss_fn import nll_loss

LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")
node_num = int(sys.argv[1])

# Model
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


async def fit_model_on_worker(
    worker,
    built_model: sy.Plan,
    built_loss_fn: sy.Plan,
    encrypters,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
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
    num_of_parameters = len(built_model.parameters())
    built_model.id = "GlobalModel"
    # built_loss_fn.id = "LossFunc"
    # model_config = sy.ModelConfig(model=built_model,
    #                           loss_fn=built_loss_fn,
    #                           optimizer="SGD",
    #                           batch_size=batch_size,
    #                           optimizer_args={"lr": lr},
    #                           epochs=1,
    #                           max_nr_batches=max_nr_batches)
    # model_config_send_start = time.time()
    built_model.send(worker)
    # model_config_send_end = time.time()
    print("[trace] GlobalInformationSend duration", worker.id, model_config_send_end - model_config_send_start)

    return_ids = [0, 1]
    for i in range(num_of_parameters):
        return_ids.append("p" + str(i))

    fit_sagg_start = time.time()
    result_list = await worker.async_fit_sagg_mc(dataset_key="mnist", encrypters=encrypters, return_ids=return_ids)
    fit_sagg_end = time.time()
    print("[trace] FitSagg", "duration", worker.id, fit_sagg_end - fit_sagg_start)

    loss = result_list[0]
    num_of_training_data = result_list[1]
    enc_params = result_list[2:]

    print("Iteration %s: %s loss: %s" % (curr_round, worker.id, loss))

    return worker.id, enc_params, loss, num_of_training_data


async def main():
    hook = sy.TorchHook(torch)
    device = torch.device("cpu")
    model = Net()
    model.build(torch.zeros([1, node_num], dtype=torch.float).to(device))

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

    alice = NodeClient(hook, "ws://10.0.17.6:6666" , id="flvm-2")
#     bob = NodeClient(hook, "ws://172.16.179.22:6667" , id="bob")
#     charlie = NodeClient(hook, "ws://172.16.179.23:6668", id="charlie")
#     med24 = NodeClient(hook, "ws://172.16.179.24:6669", id="med24")
#     testing = NodeClient(hook, "ws://localhost:6669" , id="testing")

    # worker_list = [alice, bob, charlie]
    worker_list = [alice]
    grid = sy.PrivateGridNetwork(*worker_list)

    for epoch in range(epoch_num):

        logger.info("round %s/%s", epoch, epoch_num)

        for worker in worker_list:

            built_model.id = "GlobalModel"
            # built_loss_fn.id = "LossFunc"
            # model_config = sy.ModelConfig(model=built_model,
            #                           loss_fn=built_loss_fn,
            #                           optimizer="SGD",
            #                           batch_size=batch_size,
            #                           optimizer_args={"lr": lr},
            #                           epochs=1,
            #                           max_nr_batches=-1)
            model_send_start = time.time()
            ##pdb.set_trace()
            built_model.send(worker)
            model_send_end = time.time()
            # print("[TEST]", "ModelSend", "time", model_send_start, model_send_end)
            print("[trace] ModelSend duration", worker.id, model_send_end - model_send_start)


            built_model.pointers = {}
            built_loss_fn.pointers = {}

            # decay learning rate
            learning_rate = max(0.98 * learning_rate, lr * 0.01)


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    asyncio.get_event_loop().run_until_complete(main())
#     main()

