# glaucoma-vf-ml
Deep learning for glaucoma analysis using visual field (perimetry) data

## Getting the data

```
make data
```

## Ubelix

Load conda:

```
module load Anaconda3
conda --version
```

You can always check which modules are loaded in you current shell:

```
module list
```

Create a conda env:

```
conda create -n glaucoma python=3.13
```

Initialize conda: (only need to run this once, it sets up conda in `~/.bashrc`)

```
conda init
source ~/.bashrc
```

Activate the environment:

```
conda activate glaucoma
```

Load CUDA:
```
module load CUDA/12.1
nvcc --version
```

Install PyTorch:

```
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install uv: (will be used for (non-GPU-related) Python package management)

```
pip install uv
```


