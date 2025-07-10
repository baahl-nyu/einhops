import time
import torch
import einhops

import numpy as np
from tqdm import tqdm

NUM_RUNS = 10

a = torch.randn(2,4,8,8)
b = torch.randn(1,4,4,2,8)
o = torch.einsum('pqrs,tuqvr->pstuv', [a, b])
# tests from https://rockt.ai/2018/04/30/einsum

times = []
for i in tqdm(range(NUM_RUNS)):
    a_ckks = einhops.encrypt(a, level=3)
    b_ckks = einhops.encrypt(b, level=3)
    start = time.time()
    o_ckks = einhops.einsum("pqrs,tuqvr->pstuv", a_ckks, b_ckks)
    end = time.time()
    times.append(end - start)



print(np.mean(times))
print(np.std(times))
print("validate:")
print("L2 norm:", torch.norm(einhops.decrypt(o_ckks) - o))