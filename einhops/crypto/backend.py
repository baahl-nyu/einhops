import os
import math
import torch
import psutil
import logging
import desilofhe
from desilofhe import Engine
from .backend_config import get_backend, BackendType

# need roughly 32GB of RAM for all keys: power-of-two + BSGS keys
RAM_NEEDED = 32 #GB 
logger = logging.getLogger("einhops")


def system_ram_gb():
    """Return total physical RAM in GiB."""
    return psutil.virtual_memory().total / 1024**3


def gpu_vram_gb():
    """Return total VRAM in GiB."""
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def create_engine():
    """
    Creates a Desilo FHE engine based upon the system's resources.

    """
    if torch.cuda.is_available() and gpu_vram_gb() > RAM_NEEDED:
        logger.info(f"Using GPU with {gpu_vram_gb()}GB of VRAM.")
        engine = Engine(max_level=17, mode='gpu')
    elif torch.cuda.is_available():
        logger.info(f"GPU VRAM is less than {RAM_NEEDED}GB. BSGS Keys will not be generated.")
        os.environ['EINHOPS_DISABLE_BSGS_KEYS'] = '1'
        engine = Engine(max_level=17, mode='gpu')
    else:
        engine = Engine(max_level=17, mode='parallel', thread_count=psutil.cpu_count(logical=False))
    return engine


print("Creating CKKS context...")
engine = create_engine()
SLOT_COUNT = engine.slot_count
MAX_LEVEL = engine.max_level
assert SLOT_COUNT == 16384, "CKKS slot count must be 16384"

print("Generating keys...")
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
relinearization_key = engine.create_relinearization_key(secret_key)
rotation_key = engine.create_rotation_key(secret_key)


fixed_rotation_keys = {}
keys_initialized = False

def select_bsgs_factors(n):
    """
    Selects the BSGS factors for the given number of slots. N1, N2 are O(sqrt(n))

    Args:
        n (int): The number of slots (a power of two)

    Returns:
        tuple (int, int): The BSGS factors such that n = N1 * N2.
    """

    power = int(math.log2(n))
    N1 = int(math.sqrt(2*n if power % 2 else n))
    N2 = n // N1
    return (N1, N2)  


def generate_bsgs_keys():
    """Generate all keys required for BSGS."""
    global fixed_rotation_keys, keys_initialized
    if keys_initialized:
        logger.info("BSGS keys already generated.")
        return

    if system_ram_gb() < RAM_NEEDED:
        logger.error("System RAM is less than 30GB. Only power of 2 keys are generated.")
        return

    logger.info("Generating BSGS keys...")
    N1, N2 = select_bsgs_factors(SLOT_COUNT)
    maximal_set = [-i for i in range(N1)] + [-j*N1 for j in range(N2)]
    fixed_rotation_keys = {i : engine.create_fixed_rotation_key(secret_key, i) for i in maximal_set}
    keys_initialized = True
    logger.info("Complete.")


# generate keys if not disabled
if not os.getenv('EINHOPS_DISABLE_BSGS_KEYS'):
    generate_bsgs_keys()


def fhe_encode(values):
    """
    Cleartext -> Plaintext. Input MUST be a flattened 1-d vector.
    """

    assert len(values) <= SLOT_COUNT, f"You must encode 1-d vectors of size {SLOT_COUNT} or smaller"
    backend = get_backend()
    if backend == BackendType.TORCH:
        return values
    return engine.encode(values)


def fhe_decode(values):
    """
    Plaintext -> Cleartext.
    """
    backend = get_backend()
    if backend == BackendType.TORCH:
        return values
    return engine.decode(values)


def fhe_encrypt(plaintext, level=MAX_LEVEL):
    """
    Plaintext -> Ciphertext.
    """
    backend = get_backend()
    if backend == BackendType.TORCH:
        return plaintext
    return engine.encrypt(plaintext, public_key, level)


def fhe_decrypt(ciphertext):
    """
    Ciphertext -> Plaintext.
    """
    backend = get_backend()
    if backend == BackendType.TORCH:
        return ciphertext
    return engine.decrypt_to_plaintext(ciphertext, secret_key)


def fhe_level_down(ciphertext, level):
    """
    Ciphertext -> Ciphertext (mod down to `level`)
    """
    backend = get_backend()
    if backend == BackendType.TORCH:
        return ciphertext
    return engine.level_down(ciphertext, level)


def fhe_add(op1, op2):
    """
    SIMD Addition.
    """
    backend = get_backend()
    if backend == BackendType.TORCH:
        return op1 + op2

    if isinstance(op1, torch.Tensor):
        op1 = fhe_encode(op1)
    if isinstance(op2, torch.Tensor):
        op2 = fhe_encode(op2)
    return engine.add(op1, op2)


def fhe_mul(op1, op2):
    """
    SIMD Multiplication.
    """
    backend = get_backend()
    if backend == BackendType.TORCH:
        return op1 * op2

    ## rescale first -> check here: https://fhe.desilo.dev/latest/quickstart/#seal-style-api
    if isinstance(op1, desilofhe.Ciphertext) and isinstance(op2, desilofhe.Ciphertext):
        op1 = engine.rescale(op1)
        op2 = engine.rescale(op2)
        return engine.multiply(op1, op2, relinearization_key)
    elif isinstance(op1, desilofhe.Ciphertext) and isinstance(op2, torch.Tensor):
        op1 = engine.rescale(op1)
        return engine.multiply(op1, op2)
    else:
        op2 = engine.rescale(op2)
        return engine.multiply(op1, op2)


def fhe_rotate(op1, delta):
    """
    Cyclic Rotation. Uses power-of-two rotation keys to achieve any delta. 
    """
    backend = get_backend()
    if backend == BackendType.TORCH:
        return torch.roll(op1, delta)
    
    return engine.rotate(op1, rotation_key, delta)


def fhe_rotate_fixed(op1, delta):
    """
    Rotate a ciphertext with one-key switch if keys are initialized.
    """
    backend = get_backend()
    if backend == BackendType.TORCH:
        return torch.roll(op1, delta)

    if keys_initialized:
        return engine.rotate(op1, fixed_rotation_keys[delta])
    else:
        return engine.rotate(op1, rotation_key, delta)


def fhe_hoisted_rotate(op1, deltas):
    """
    Rotate a ciphertext by different amounts.
    Single-hoisting (only one ModUp rather than len(deltas) ModUps) if keys are initialized.
    """

    backend = get_backend()
    if backend == BackendType.TORCH:
        return [torch.roll(op1, -delta) for delta in deltas]
    
    if keys_initialized:
        return engine.rotate_batch(op1, [fixed_rotation_keys[-delta] for delta in deltas])
    else:
        return [engine.rotate(op1, rotation_key, -delta) for delta in deltas]
