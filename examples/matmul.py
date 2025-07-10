import torch
import einhops

a = torch.randn(2,5)
b = torch.randn(5,3)
o = torch.einsum('ij,jk->ik', a, b)

a_ckks = einhops.encrypt(a, level=3)
b_ckks = einhops.encrypt(b, level=3)

o_ckks = einhops.einsum("ij,jk->ik", a_ckks, b_ckks)

print('torch:')
print(o)
print('einhops:')
print(einhops.decrypt(o_ckks))
