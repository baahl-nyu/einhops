import time
import torch
import einhops

import numpy as np
from tqdm import tqdm

NUM_RUNS = 10

a = torch.randn(16, 8)
b = torch.randn(8, 8)
c = torch.randn(8, 16)
o = torch.einsum('ij,jk,kl->il', a, b, c)
# tests from https://rockt.ai/2018/04/30/einsum

times = []
for i in tqdm(range(NUM_RUNS)):
    a_ckks = einhops.encrypt(a, level=4)
    b_ckks = einhops.encrypt(b, level=4)
    c_ckks = einhops.encrypt(c, level=4)
    start = time.time()
    o_ckks = einhops.einsum("ij,jk,kl->il", a_ckks, b_ckks, c_ckks)
    end = time.time()
    times.append(end - start)



print(np.mean(times))
print(np.std(times))
print("validate:")
print("L2 norm:", torch.norm(einhops.decrypt(o_ckks).cpu() - o))