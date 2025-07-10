from enum import Enum

class BackendType(Enum):
    CKKS = "ckks"
    TORCH = "torch"

_backend = BackendType.CKKS

def set_backend(backend):
    global _backend
    mapping = {'ckks': BackendType.CKKS, 'torch': BackendType.TORCH}
    backend = backend.lower()
    if backend not in mapping:
        raise ValueError(f"Backend must be a BackendType (ckks or torch), got {backend}")
    _backend = mapping[backend]

def get_backend():
    return _backend