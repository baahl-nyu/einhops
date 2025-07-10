import logging
from .crypto.backend_config import BackendType, set_backend, get_backend
from .crypto.ckkstensor import CKKSTensor, tensor_to_fhe, fhe_to_tensor
from .crypto.backend import MAX_LEVEL
from .engine import EinsumEngine


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False
_engine = EinsumEngine()


def encrypt(tensor, level=MAX_LEVEL):
    """
    Encrypts a tensor.

    Args:
        tensor (torch.Tensor): The tensor to encrypt.
        level (int): The level to encrypt the tensor to.

    Returns:
        CKKSTensor: The encrypted tensor.
    """

    logger.debug("encrypting tensor")
    return tensor_to_fhe(tensor, level)


def decrypt(ckkstensor):
    """
    Decrypts a tensor.

    Args:
        ckkstensor (CKKSTensor): The encrypted tensor to decrypt.

    Returns:
        torch.Tensor: The decrypted tensor.
    """

    logger.debug("decrypting tensor")
    return fhe_to_tensor(ckkstensor)


def einsum(equation, *args):
    """
    Performs an einsum operation.

    Args:
        equation (str): The equation to perform.
        *args: The tensors to perform the operation on (either CKKSTensors or torch.Tensors)

    Returns:
        CKKSTensor: The result of the einsum operation.
    """

    return _engine.einsum(equation, *args)


def set_log_level(level):
    """
    Sets the log level.

    Args:
        level (str): DEBUG, INFO, WARNING, ERROR, CRITICAL
    """

    logger.setLevel(level)


__all__ = [
    'encrypt',
    'decrypt',
    'einsum',
    'CKKSTensor',
    'set_log_level',
    'BackendType',
    'set_backend',
    'get_backend',
]
