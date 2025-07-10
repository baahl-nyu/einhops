import os; os.environ['EINHOPS_DISABLE_BSGS_KEYS'] = '1'
import torch
import einhops


def test_three_way_hadamard_mult():
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    c = torch.randn(2, 3, dtype=torch.float32)
    o = torch.einsum('ij,ij,ij->ij', a, b, c)
    a_ckks = einhops.encrypt(a)
    b_ckks = einhops.encrypt(b)
    c_ckks = einhops.encrypt(c)
    o_ckks = einhops.einsum("ij,ij,ij->ij", a_ckks, b_ckks, c_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


def test_chained_matmatmul():
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    c = torch.randn(4, 2, dtype=torch.float32)
    o = torch.einsum('ij,jk,kl->il', a, b, c)
    a_ckks = einhops.encrypt(a)
    b_ckks = einhops.encrypt(b)
    c_ckks = einhops.encrypt(c)
    o_ckks = einhops.einsum("ij,jk,kl->il", a_ckks, b_ckks, c_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


def test_bilinear_transform():
    a = torch.randn(2,3, dtype=torch.float32)
    b = torch.randn(5,3,7, dtype=torch.float32)
    c = torch.randn(2,7, dtype=torch.float32)
    o = torch.einsum('ik,jkl,il->ij', [a, b, c])
    a_ckks = einhops.encrypt(a)
    b_ckks = einhops.encrypt(b)
    c_ckks = einhops.encrypt(c)
    o_ckks = einhops.einsum("ik,jkl,il->ij", a_ckks, b_ckks, c_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


def test_tensor_contraction():
    a = torch.randn(2,3,5,7, dtype=torch.float32)
    b = torch.randn(1,4,3,2,5, dtype=torch.float32)
    o = torch.einsum('pqrs,tuqvr->pstuv', [a, b])
    a_ckks = einhops.encrypt(a)
    b_ckks = einhops.encrypt(b)
    o_ckks = einhops.einsum("pqrs,tuqvr->pstuv", a_ckks, b_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)