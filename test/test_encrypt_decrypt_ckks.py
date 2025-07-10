import os; os.environ['EINHOPS_DISABLE_BSGS_KEYS'] = '1'
import torch
import einhops


def test_encrypt_decrypt_0d():
    a = torch.empty(())
    b = einhops.decrypt(einhops.encrypt(a))
    assert torch.allclose(a, b, atol=1e-4)


def test_encrypt_decrypt_1d():
    a = torch.randn(10)
    b = einhops.decrypt(einhops.encrypt(a))
    assert torch.allclose(a, b, atol=1e-4)


def test_encrypt_decrypt_2d():
    a = torch.randn(10, 10)
    b = einhops.decrypt(einhops.encrypt(a))
    assert torch.allclose(a, b, atol=1e-4)


def test_encrypt_decrypt_3d():
    a = torch.randn(3, 4, 5)
    b = einhops.decrypt(einhops.encrypt(a))
    assert torch.allclose(a, b, atol=1e-4)


def test_encrypt_decrypt_4d():
    a = torch.randn(3, 4, 5, 6)
    b = einhops.decrypt(einhops.encrypt(a))
    assert torch.allclose(a, b, atol=1e-4)

