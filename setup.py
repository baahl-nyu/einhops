from setuptools import setup, find_packages

setup(
    name="einhops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==2.3.1",
        "opt_einsum==3.4.0",
        "psutil==7.0.0",
        "torch==2.7.1",
        "tqdm==4.67.1",
        "pytest==8.4.1",
        "pytest-cov==6.2.1",
    ],
    extras_require={
        "cpu": ["desilofhe"],
        "cuda121": ["desilofhe-cu121"],
        "cuda124": ["desilofhe-cu124"],
        "cuda126": ["desilofhe-cu126"],
        "cuda128": ["desilofhe-cu128"],
        "cuda129": ["desilofhe-cu129"],
        "cuda130": ["desilofhe-cu130"],
    },
    python_requires=">=3.11",
    description="Einsum notation for expressive homomorphic operations on RNS-CKKS tensors",
    author="Karthik Garimella",
    author_email="kg2383@nyu.edu",
)
