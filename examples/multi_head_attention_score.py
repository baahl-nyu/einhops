import time
import torch
import einhops

einhops.set_log_level('DEBUG')

# batch_size, seq_len, n_heads, h_dim_per_head
q = torch.randn(2, 5, 8, 16)
k = torch.randn(2, 5, 8, 16)
attn = torch.einsum("bthd,bThd->bhtT", q, k)

q_ckks = einhops.encrypt(q, level=3)
k_ckks = einhops.encrypt(k, level=3)

attn_ckks = einhops.einsum("bthd,bThd->bhtT", q_ckks, k_ckks)

print('validate:')
print("L2 norm: ", torch.norm(einhops.decrypt(attn_ckks) - attn))