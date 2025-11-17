# EinHops: Einsum Notation for Expressive Homomorphic Operations on RNS-CKKS Tensors

EinHops enables the intuitive [einsum notation](https://ejenner.com/post/einsum/) for performing tensor operations directly on encrypted tensors in Fully Homomorphic Encryption (FHE). The example below demonstrates batched matrix-matrix multiplication on encrypted data.

```python
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
```

## What is EinHops?
EinHops ia a system for performing tensor operations via einsum expressions in FHE (RNS-CKKS). The high level goal of EinHops is to build a *simple* packing strategy that provides transparency into how your data is arranged within ciphertext slots and provide a minimalist implementation. We decompose einsum expressions into a series of FHE-friendly operations and implement each step directly in Python. For more information, check out our [paper](https://arxiv.org/abs/2507.07972)!

## Requirements
- Python 3.11 or greater
- 3GB RAM (low memory mode) or 32GB RAM (full BSGS keys)
- (Optional) NVIDIA GPU with CUDA 12.x

We've tested EinHops on an Intel Xeon Platinum 8480+ processor with an H100 80GB, an AMD EPYC 7502 32-Core Processor with an RTX 3090, and a Macbook Air with an Apple M2 chip.

## Installation
Clone the repo and change directories:
```bash
git clone https://github.com/baahl-nyu/einhops.git
cd einhops
```
We recommend installing EinHops in a virtual environment using Python version 3.11. We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/), but `conda` or `pyenv` would also work. For `uv`, you can run the following in the `einhops` repo:
```bash
uv venv --python 3.11
source .venv/bin/activate
```
You should now see `(einhops)` at the beginning of your shell prompt. Next, we will make sure to install the correct [Desilo FHE library](https://fhe.desilo.dev/latest/install/) version based on your system setup.

### CPU Installation
If you wish to install EinHops to run on your CPU, run:
```bash
uv pip install -e."[cpu]"
```

### GPU Installation
If you have an NVIDIA GPU and CUDA version >=12.1, you can install the appropriate Desilo version by checking your CUDA version (top right corner of `nvidia-smi`). More details on Desilo GPU versions can be found [here](https://fhe.desilo.dev/latest/install/). In our case, we have version 12.8 so we run:
```bash
uv pip install -e ."[cuda128]"  # or cuda121, cuda124, cuda126, cuda129, cuda130
```

You should now be able to run the examples within the `examples` folder.

## Tests
The following will run all test cases:
```bash
pytest test/ --cov=einhops
```

## Running EinHops
There are a couple of configurations to help you run EinHops under different circumstances:
### Low Memory Environment
If you are constrained on RAM or VRAM, you can choose to generate only the power-of-two rotations keys which requires roughly 3GB of memory. Set the following environment flag:
```bash
EINHOPS_DISABLE_BSGS_KEYS=1
```
This mode will be slightly slower, but otherwise EinHops will generate the rotation keys needed for baby-step giant-step linear transformations, which will require roughly 32GB of memory.
### Verbosity
You can set the verbosity level through the logger.
```python
import einhops
einhops.set_log_level("DEBUG") # DEBUG -> INFO -> WARNING (default) -> ERROR -> CRITICAL
```
Setting to DEBUG level will help with understanding the dataflow for each stage of the einsum call listed in the paper.

### Simulated CKKS Backend
You can also replace the CKKS backend with PyTorch to help debug and understand the dataflow of FHE programs you build in EinHops. This will still execute the same dataflow but with torch Tensors rather than ciphertexts.
```python
import einhops
einhops.set_backend("torch") # torch or ckks (default)
```

## Fun Examples
### Sum over a dimension
```python
import torch
import einhops
a = torch.randn(128, 128)
o = torch.einsum("ij->j", a)

a_ckks = einhops.encrypt(a)
o_ckks = einhops.einsum("ij->j", a_ckks)

print(o)
print(einhops.decrypt(o_ckks))
```
### Multi-Head Self-Attention Scores
```python
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
```

## Citation
If you found our work useful, please use the following citation:
```bash
@misc{garimella2025einhopseinsumnotationexpressive,
      title={EinHops: Einsum Notation for Expressive Homomorphic Operations on RNS-CKKS Tensors},
      author={Karthik Garimella and Austin Ebel and Brandon Reagen},
      year={2025},
      eprint={2507.07972},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2507.07972},
}
```

## Issues
Please feel free to open an issue or make a pull request!
