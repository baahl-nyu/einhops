import torch
import einhops

a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)
c = torch.einsum("bik,bkj->bij", a, b)

a_ctxt = einhops.encrypt(a)
b_ctxt = einhops.encrypt(b)
c_ctxt = einhops.einsum("bik,bkj->bij", a_ctxt, b_ctxt)

assert c.shape == c_ctxt.shape == (2, 3, 5)
assert torch.allclose(c, einhops.decrypt(c_ctxt), atol=1e-4)
