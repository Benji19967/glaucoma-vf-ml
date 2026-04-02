# glaucoma-vf-ml
Deep learning for glaucoma analysis using visual field (perimetry) data

## Folder Structure

    .
    │
    ├── data
    │   ├── GRAPE               # GRAPE dataset
    │   └── UWHVF               # UWHVF dataset
    │   
    ├── docs                    # Model card for each model
    │   
    ├── logs
    │   ├── <model_1>
    │       ├── <version_1>     # Logs, configs, and checkpoints for <model> & <version>
    │       └── <version_2>
    │   ├── <model_2>
    │       └── ...
    │   └── ...
    │   
    ├── notebooks
    │   ├── GRAPE.ipynb         # Exploring the GRAPE dataset
    │   └── UWHVF.ipynb         # Exploring the UWHVF dataset
    │   
    ├── scripts
    │   ├── setup_data.py       # Download the GRAPE and UWHVF datasets
    │   ├── test.sh             # Script to launch test run
    │   └── train.sh            # Script to launch train run
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
| [Glaucoma classification](./docs/task1_hvf_system.md) | CNN | 54-point HVF | in progress |

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

### Training a model (`fit`)

Configure the params in `config/<model_name>.yaml` (create the file when adding new models).

Training the default model:
```
./scripts/train.sh
```

Training a specific model:
```
./scripts/train.sh <model_name>
```

These commands will create an entry in `logs/<model_name>/<version>`. The `version` is `v_<timestamp>`. 


### Testing a model (`test`)

Testing the default model:
```
./scripts/test.sh
```

Testing the latest version of a model:
```
./scripts/test.sh <model_name>
```

Testing a specific version of a model:
```
./scripts/test.sh <model_name> <version_name>
```
For example
```
./scripts/test.sh hvf_system v_20260331_122923
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
