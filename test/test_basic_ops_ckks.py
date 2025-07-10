import os; os.environ['EINHOPS_DISABLE_BSGS_KEYS'] = '1'
import torch
import einhops


def test_matrix_transpose():
    a = torch.arange(6, dtype=torch.float32).reshape(2,3)
    o = torch.einsum("ij->ji", a)
    a_ckks = einhops.encrypt(a)
    o_ckks = einhops.einsum("ij->ji", a_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


def test_sum():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    o = torch.einsum('ij->', [a])
    a_ckks = einhops.encrypt(a)
    o_ckks = einhops.einsum("ij->", a_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


def test_column_sum():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    o = torch.einsum('ij->j', [a])
    a_ckks = einhops.encrypt(a)
    o_ckks = einhops.einsum("ij->j", a_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


def test_row_sum():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    o = torch.einsum('ij->i', [a])
    a_ckks = einhops.encrypt(a)
    o_ckks = einhops.einsum("ij->i", a_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)


def test_dot_product():
    a = torch.arange(3, dtype=torch.float32)
    b = torch.arange(3,6, dtype=torch.float32)
    o = torch.einsum('i,i->', [a, b])
    a_ckks = einhops.encrypt(a)
    b_ckks = einhops.encrypt(b)
    o_ckks = einhops.einsum("i,i->", a_ckks, b_ckks)
    assert torch.allclose(o, einhops.decrypt(o_ckks), atol=1e-4)
