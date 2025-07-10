import os; os.environ['EINHOPS_DISABLE_BSGS_KEYS'] = '1'
import torch
import einhops


einhops.set_backend("torch")


# neither inputs are encrypted
def test_matvec_mult0():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(3, dtype=torch.float32)
    o = torch.einsum('ik,k->i', [a, b])
    o_ckks = einhops.einsum("ik,k->i", a, b)
    assert torch.allclose(o, o_ckks, atol=1e-4)


# op1 is encrypted
def test_matvec_mult1():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(3, dtype=torch.float32)
    o = torch.einsum('ik,k->i', [a, b])
    a_ckks = einhops.encrypt(a)
    o_ckks = einhops.einsum("ik,k->i", a_ckks, b)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


# op2 is encrypted
def test_matvec_mult2():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(3, dtype=torch.float32)
    o = torch.einsum('ik,k->i', [a, b])
    b_ckks = einhops.encrypt(b)
    o_ckks = einhops.einsum("ik,k->i", a, b_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


# both ops are encrypted
def test_matvec_mult3():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(3, dtype=torch.float32)
    o = torch.einsum('ik,k->i', [a, b])
    a_ckks = einhops.encrypt(a)
    b_ckks = einhops.encrypt(b)
    o_ckks = einhops.einsum("ik,k->i", a_ckks, b_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


# op1 is encrypted
def test_matmat_mult1():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    o = torch.einsum('ik,kj->ij', [a, b])
    a_ckks = einhops.encrypt(a)
    o_ckks = einhops.einsum("ik,kj->ij", a_ckks, b)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


# op2 is encrypted
def test_matmat_mult2():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    o = torch.einsum('ik,kj->ij', [a, b])
    b_ckks = einhops.encrypt(b)
    o_ckks = einhops.einsum("ik,kj->ij", a, b_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


# both ops are encrypted
def test_matmat_mult3():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    o = torch.einsum('ik,kj->ij', [a, b])
    a_ckks = einhops.encrypt(a)
    b_ckks = einhops.encrypt(b)
    o_ckks = einhops.einsum("ik,kj->ij", a_ckks, b_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)