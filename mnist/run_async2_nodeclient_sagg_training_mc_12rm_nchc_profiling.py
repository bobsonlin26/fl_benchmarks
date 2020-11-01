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

from syft.frameworks.torch.fl.loss_fn import nll_loss

LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")

# @torch.jit.script
# def loss_fn(pred, target):
#     return F.nll_loss(input=pred, target=target)

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

# Model
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
    built_model.id = "GlobalModel_MNIST"
    built_loss_fn.id = "LossFunc"
    model_config = sy.ModelConfig(model=built_model,
                              loss_fn=built_loss_fn,
                              optimizer="SGD",
                              batch_size=batch_size,
                              optimizer_args={"lr": lr},
                              epochs=1,
                              max_nr_batches=max_nr_batches)
    # model_config_send_start = time.time()
    # pdb.set_trace()
    # model_config.send(worker)
    # model_config_send_end = time.time()
    # print("[trace] GlobalInformationSend duration", worker.id, model_config_send_end - model_config_send_start)

    return_ids = [0, 1]
    for i in range(num_of_parameters):
        return_ids.append("p" + str(i))

    fit_sagg_start = time.time()
    result_list = await worker.async_fit2_sagg_mc(model_config, dataset_key="mnist", encrypters=encrypters, return_ids=return_ids)
    fit_sagg_end = time.time()
    print("[trace] FitSagg", "duration", worker.id, fit_sagg_end - fit_sagg_start)

    loss = result_list[0]
    num_of_training_data = result_list[1]
    enc_params = result_list[2:]

    print("Iteration %s: %s loss: %s" % (curr_round, worker.id, loss))

    return worker.id, enc_params, loss, num_of_training_data


def evaluate_model_on_worker(
    model_identifier,
    worker,
    dataset_key, model, built_loss_fn,
    nr_bins,
    batch_size,
    device,
    print_target_hist=False):

    model_config = sy.ModelConfig(
        batch_size=batch_size, model=model, loss_fn=built_loss_fn, optimizer_args=None, epochs=1
    )
    model_config.send(worker)

    result = worker.evaluate_mc(
        dataset_key=dataset_key,
        return_histograms=True,
        nr_bins=nr_bins,
        return_loss=True,
        return_raw_accuracy=True,
        device=device,
    )

    test_loss = result["loss"]
    correct = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]
    hist_pred = result["histogram_predictions"]
    hist_target = result["histogram_target"]

    if print_target_hist:
        logger.info("Target histogram: %s", hist_target)
    percentage_0_3 = int(100 * sum(hist_pred[0:4]) / len_dataset)
    percentage_4_6 = int(100 * sum(hist_pred[4:7]) / len_dataset)
    percentage_7_9 = int(100 * sum(hist_pred[7:10]) / len_dataset)
    logger.info(
        "%s: Percentage numbers 0-3: %s%%, 4-6: %s%%, 7-9: %s%%",
        model_identifier,
        percentage_0_3,
        percentage_4_6,
        percentage_7_9,
    )

    logger.info(
        "%s: Average loss: %s, Accuracy: %s/%s (%s%%)",
        model_identifier,
        f"{test_loss:.4f}",
        correct,
        len_dataset,
        f"{100.0 * correct / len_dataset:.2f}",
    )

async def main():
    hook = sy.TorchHook(torch)
    device = torch.device("cpu")
    model = Net()
    model.build(torch.zeros([1, 1, 28, 28], dtype=torch.float).to(device))

    @sy.func2plan()
    def loss_fn(pred, target):
        return nll_loss(input=pred, target=target)

    input_num = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    dummy_pred = F.log_softmax(input_num, dim=1)
    loss_fn.build(dummy_pred, target)

    epoch_num = 21
    batch_size = 64
    lr = 0.1
    learning_rate = lr
    optimizer_args = {"lr" : lr}

    # alice = NodeClient(hook, "ws://172.16.179.20:6666" , id="alice")
    # bob = NodeClient(hook, "ws://172.16.179.21:6667" , id="bob")
    # charlie = NodeClient(hook, "ws://172.16.179.22:6668", id="charlie")
#     testing = NodeClient(hook, "ws://localhost:6669" , id="testing")

    # worker_list = [alice, bob, charlie]

    worker_list = []
    for i in range(2, 14):
        worker = NodeClient(hook, "ws://"+flvm_ip[i]+":6666" , id="flvm-"+str(i))
        worker_list.append(worker)


    grid = sy.PrivateGridNetwork(*worker_list)

    for epoch in range(epoch_num):

        logger.info("Training round %s/%s", epoch, epoch_num)

        round_start_time = time.time()

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=worker,
                    built_model=model,
                    built_loss_fn=loss_fn,
                    encrypters=worker_list,
                    batch_size=batch_size,
                    curr_round=epoch,
                    max_nr_batches=-1,
                    lr=0.1,
                )
                for worker in worker_list
            ]
        )

        local_train_end_time = time.time()
        print("[trace]", "AllWorkersTrainingTime", "duration", "COORD", local_train_end_time - round_start_time)

        enc_models = {}
        loss_values = {}
        data_amounts = {}
        total_data_amount = 0


        for worker_id, enc_params, worker_loss, num_of_training_data in results:
            if enc_params is not None:
                enc_models[worker_id] = enc_params
                loss_values[worker_id] = worker_loss
                data_amounts[worker_id] = num_of_training_data
                total_data_amount += num_of_training_data

        ## aggregation
        nr_enc_models = len(enc_models)
        enc_models_list = list(enc_models.values())
        data_amounts_list = list(data_amounts.values()) ##
        dst_enc_model = enc_models_list[0]

        aggregation_start_time = time.time()
        with torch.no_grad():
            for i in range(len(dst_enc_model)):
                for j in range(1, nr_enc_models):
                    dst_enc_model[i] += enc_models_list[j][i]
        aggregation_end_time = time.time()
        print("[trace]", "AggregationTime", "duration", "COORD", aggregation_end_time - aggregation_start_time)


        ## decryption
        new_params = []
        decryption_start_time = time.time()
        with torch.no_grad():
            for i in range(len(dst_enc_model)):
                decrypt_para = dst_enc_model[i].get()
                new_para = decrypt_para.float_precision()
                new_para = new_para / int(total_data_amount)
                model.parameters()[i].set_(new_para)

        round_end_time = time.time()
        print("[trace]", "DecryptionTime", "duration", "COORD", round_end_time - decryption_start_time)
        print("[trace]", "RoundTime", "duration", "COORD", round_end_time - round_start_time)

        ## FedAvg
#         nr_models = len(models)
#         model_list = list(models.values())
#         dst_model = model_list[0]
#         nr_params = len(dst_model.parameters())
#         with torch.no_grad():
#             for i in range(1, nr_models):
#                 src_model = model_list[i]
#                 src_params = src_model.parameters()
#                 dst_params = dst_model.parameters()
#                 for i in range(nr_params):
#                     dst_params[i].set_(src_params[i].data + dst_params[i].data)
#             for i in range(nr_params):
#                 dst_params[i].set_(dst_params[i].data * 1/total_data_amount)


#         if epoch%5 == 0 or epoch == 49:
#             evaluate_model_on_worker(
#                 model_identifier="Federated model",
#                 worker=testing,
#                 dataset_key="mnist_testing",
#                 model=model,
#                 built_loss_fn=loss_fn,
#                 nr_bins=10,
#                 batch_size=64,
#                 device=device,
#                 print_target_hist=False,
#             )

        model.pointers = {}
        loss_fn.pointers = {}

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

