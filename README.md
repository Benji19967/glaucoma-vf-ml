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

## Running locally

### Setting up the python `.venv`

```
uv sync
```

### Getting the data

```
source .venv/bin/activate
python scripts/setup_data.py
```

### Running the training

```
python src/glaucoma_vf/training/train.py
```


### Checking the logs

```
tensorboard --logdir .
```

## Running on Ubelix HPC (with GPUs)

### Building the apptainer

```
apptainer build glaucoma-ml.sif Apptainer.def
```

### Running the apptainer

To run on a GPU node:

```
srun --partition=gpu --gres=gpu:rtx4090:1 --cpus-per-task=4 --nodes=1 --mem=16G --time=02:00:00 --pty bash
```

then on the `gnode`:

```
apptainer run --nv --bind ./data:/glaucoma-vf-ml/data glaucoma-ml.sif
```


### Creating a sandbox apptainer environment

This is useful for quick development and avoids having to rebuild the `.sif` file from scratch every time.

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

Now you can install (`apt` or `python`) packages from within the container to test things out. 
Make sure to edit the `Apptainer.def` file for permanent changes.

Note: `which` may not work as expected inside the apptainer, use `command -v` instead.