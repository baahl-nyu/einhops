import time
import torch
import einhops

import numpy as np
from tqdm import tqdm

NUM_RUNS = 10

a = torch.randn(128, 128)
o = torch.einsum("ij->", a)

# tests from https://rockt.ai/2018/04/30/einsum

times = []
for i in tqdm(range(NUM_RUNS)):
    a_ckks = einhops.encrypt(a, level=3)
    start = time.time()
    o_ckks = einhops.einsum("ij->", a_ckks)
    end = time.time()
    times.append(end - start)



print(np.mean(times))
print(np.std(times))
print("validate:")
print("L2 norm:", torch.norm(einhops.decrypt(o_ckks).cpu() - o))