from syft.workers.node_client import NodeClient
import logging
import sys
import asyncio
import torch.nn as nn
import torch.nn.functional as F
import time
import pdb

from syft.workers import websocket_client
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

hook = sy.TorchHook(torch)

alice = NodeClient(hook, "ws://172.16.179.20:6666" , id="alice")
bob = NodeClient(hook, "ws://172.16.179.21:6667" , id="bob")
charlie = NodeClient(hook, "ws://172.16.179.22:6668", id="charlie")

num_a = torch.ones([node_num])
num_a = num_a * 3
fix_a = num_a.fix_precision()

num_b = torch.ones([node_num])
num_b = num_b * 4
fix_b = num_b.fix_precision()

## encrypt
enc_a = fix_a.share(alice, bob, charlie)
enc_b = fix_b.share(alice, bob, charlie)

start_time = time.time()
# pdb.set_trace()
enc_c = enc_a + enc_b
print("[PROF]", "AggTime", "duration", time.time() - start_time)

num_c = enc_c.get()
num_c = num_c.float_precision()

# print(num_c)
print("Success !")
