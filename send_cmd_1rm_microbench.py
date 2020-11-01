import syft as sy
import torch
from syft.workers.node_client import NodeClient
import time

hook = sy.TorchHook(torch)

num = torch.tensor([1])
alice = NodeClient(hook, "ws://10.0.17.6:6666" , id="flvm-2")

for i in range(11):

    start_time = time.time()
    ptr_num = num.send(alice)
    end_time = time.time()

    print("Duration:", end_time - start_time)
