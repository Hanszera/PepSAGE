# Local Frame Struct Tokenizer Training

This directory contains the code for training a local-frame structure tokenizer. The training entry point is `train.py`, and the default configuration file is `configs/train_local_tokenizer.yaml`.

## 1. Enter the Training Directory

It is recommended to always start training from this directory to avoid default configuration path resolution issues:

```powershell
cd E:\HNPHD\PepSAGE\local_frame_struct_tokenizer
```

Then run:

```powershell
python train.py --config configs\train_local_tokenizer.yaml
```

If you run the command from the repository root, you need to explicitly pass the full configuration path:

```powershell
cd E:\HNPHD\PepSAGE
python local_frame_struct_tokenizer\train.py --config local_frame_struct_tokenizer\configs\train_local_tokenizer.yaml
```

## 2. Environment Dependencies

The training code depends on PyTorch, PyTorch Lightning, OmegaConf, pandas, TensorBoard, and Mamba SSM related dependencies. This repository currently does not provide a pinned dependency file, so make sure these packages are installed before running training.

Minimal dependency checks:

```powershell
python -c "import torch, pytorch_lightning, omegaconf, pandas; print('basic deps ok')"
python -c "import mamba_ssm; print('mamba_ssm ok')"
```

If any dependency is missing, install the corresponding package in the training environment first. For GPU training, also make sure that the PyTorch, CUDA, and `mamba_ssm` versions are compatible with each other.

## 3. Data Paths

The data paths in the default configuration are resolved relative to the repository root, `E:\HNPHD\PepSAGE`.

Default Peptide/PepSAGE data locations:

```text
E:\HNPHD\PepSAGE\data\pepsage\crude
E:\HNPHD\PepSAGE\data\pepsage\processed
```

Default general protein data location:

```text
E:\HNPHD\PepSAGE\data\general_protein
```

If your data is stored elsewhere, modify the corresponding fields in `configs/train_local_tokenizer.yaml`:

```yaml
data:
  pepsage:
    train:
      structure_dir: data/pepsage/crude
      dataset_dir: data/pepsage/processed
    val:
      structure_dir: data/pepsage/crude
      dataset_dir: data/pepsage/processed
  general_protein:
    train:
      parquet_dir: null
      data_root: data/general_protein
    val:
      parquet_dir: null
      data_root: data/general_protein
```

When `parquet_dir` is not `null`, that directory is used first. Otherwise, the data path is constructed automatically from `data_root + dataset_name + split`.
