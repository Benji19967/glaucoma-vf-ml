# glaucoma-vf-ml
Deep learning for glaucoma analysis using visual field (perimetry) data

## Folder Structure

    .
    │
    ├── data
    │   ├── GRAPE               # GRAPE dataset
    │   └── UWHVF               # UWHVF dataset
    │   
    ├── notebooks
    │   ├── GRAPE.ipynb         # Exploring the GRAPE dataset
    │   └── UWHVF.ipynb         # Exploring the UWHVF dataset
    │   
    ├── scripts
    │   └── setup_data.py       # Download the GRAPE and UWHVF datasets
    │
    ├── src
    │   └── glaucoma_vf
    │       ├── models
    │       └── training
    │
    └── README.md

## Project Overview
| Task | Model | Input | Status |
| :--- | :--- | :--- | :--- |
| [Glaucoma classification](./docs/task1_classifier.md) | - | 54-point HVF | in progress |

## Setting up the python `.venv`

```
uv sync
```

## Getting the data

```
source .venv/bin/activate
python scripts/setup_data.py
```

## Running the training

```
python src/glaucoma_vf/training/train.py
```


## Checking the logs

```
tensorboard --logdir .
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

### Apptainer

#### Building the apptainer

```
apptainer build glaucoma-ml.sif Apptainer.def
```

#### Running the apptainer

To run on a GPU node:

```
srun --partition=gpu --gres=gpu:rtx4090:1 --cpus-per-task=4 --nodes=1 --mem=16G --time=02:00:00 --pty bash
```

then on the `gnode`:

```
apptainer run --nv --bind ./data:/glaucoma-vf-ml/data glaucoma-ml.sif
```

#### Creating a sandbox environment

Create a sandbox environment:

```
apptainer build --sandbox hvf_sandbox/ Apptainer.def
```

Make the sandbox editable:
```
mkdir -p hvf_sandbox/storage
mkdir -p hvf_sandbox/etc/localtime
mkdir -p hvf_sandbox/glaucoma-vf-ml/data
apptainer shell --bind data:/glaucoma-vf-ml/data --writable hvf_sandbox/
```

Alternative to `which python`: `command -v python`

