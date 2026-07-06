# Discrete Token Peptide Generation

This directory contains the pepsage training, sampling, and evaluation code. The main entry point is `train_sage.py`.

The pipeline has three stages:

1. Train or prepare a local-frame tokenizer checkpoint.
2. Train the `discret_token` / pepsage model with the frozen tokenizer.
3. Run `test_only` sampling and evaluate the generated PDB files.

## Required Checkpoints

This project depends on two different checkpoints:

- **Tokenizer checkpoint**: produced by `local_frame_struct_tokenizer`.
- **pepsage checkpoint**: produced by `discret_token/train_sage.py`.

The tokenizer checkpoint is loaded by `core/models/tokenizer_bridge.py` before pepsage training or testing. If this path is missing or points to a different checkpoint, the discrete token targets will change and the results will not be reproducible.

Example tokenizer checkpoint on this machine:

```text
PepSAGE\local_frame_struct_tokenizer\outputs\exp1_local_frame_tokenizer\compression\local_no_type_k1\pepsage\checkpoints\last.ckpt
```

Example pepsage checkpoint in this directory:

```text
PepSAGE\discret_token\output\checkpoints\last.ckpt
```

## Training

Enter the project directory:

```powershell
cd PepSAGE\discret_token
```

Run training with the default config and an explicit tokenizer checkpoint:


```bash
python train_sage.py \
  --config_file configs/default.yaml \
  --exp_name pepsage \
  --revision r0 \
  --batch_size 8 \
  --epochs 50 \
  --tokenizer_checkpoint_path /path/to/local_frame_struct_tokenizer/outputs/exp1_local_frame_tokenizer/compression/local_no_type_k1/pepsage/checkpoints/last.ckpt
```

Training outputs are written under:

```text
logs/<project_name>/<exp_name>/<revision>/
```

The saved config and model checkpoints are usually:

```text
logs/<project_name>/<exp_name>/<revision>/config.yaml
logs/<project_name>/<exp_name>/<revision>/checkpoints/last.ckpt
```

## `test_only` Sampling

The `test_only` mode loads a trained pepsage checkpoint and generates peptide samples.

Before running `test_only`, make sure the config file contains a valid tokenizer checkpoint path:

```yaml
dynamics:
  model:
    tokenizer_bridge:
      checkpoint_path: PepSAGE\local_frame_struct_tokenizer\outputs\exp1_local_frame_tokenizer\compression\local_no_type_k1\pepsage\checkpoints\last.ckpt
```


```bash
python train_sage.py --test_only \
  --config_file logs/<project_name>/<exp_name>/<revision>/config.yaml \
  --test_ckpt_path logs/<project_name>/<exp_name>/<revision>/last.ckpt \
  --num_samples 64 \
  --sample_steps 100
```

During testing:

1. The model samples trajectories and saves `.pt` files to `test_outputs...`.
2. The `ConsPep` callback converts the final samples into PDB files.
3. PDB files are saved under `generated_pep...`.

Each generated target directory should contain:

```text
gt.pdb
sample_0.pdb
sample_1.pdb
...
sample_63.pdb
```


## End-to-End Reproduction Checklist

1. Confirm the tokenizer checkpoint exists.
2. Confirm the pepsage checkpoint exists.
3. Confirm the config file points to the correct tokenizer checkpoint.
4. Run `test_only` to generate `.pt` trajectories and PDB files.
5. Run `core/callbacks/evaluate.py` for main peptide-level metrics.
6. Run `train_eval_other.py` for additional sequence and CA-distance metrics.
7. Run `evaluate_atom_metrics.py` for atom-level metrics.

Minimal local `test_only` and evaluation sequence:

```powershell

python train_sage.py ^
  --test_only ^
  --config_file output\config.yaml ^
  --test_ckpt_path last.ckpt ^
  --num_samples 64 ^
  --sample_steps 100

python core\callbacks\evaluate.py --root_dir output --num_samples 64
python train_eval_other.py --root_dir output --num_samples 64
python evaluate_atom_metrics.py --root_dir output --num_samples 64
```

## Notes

- `test_only` does not automatically run all evaluation scripts. It generates samples and PDB files; run the evaluation scripts separately unless the corresponding callback is enabled in `train_sage.py`.
- The `output/config.yaml` file may contain machine-specific absolute paths. Update them before reproducing on another machine.
- If `--tokenizer_checkpoint_path` is omitted during training, the tokenizer bridge may use a randomly initialized tokenizer, which makes the training target unreliable.
- Several scripts contain legacy server paths such as `/data10/java/CH`; update paths or run from the original environment if needed.

## Downloads

The pretrained checkpoints can be downloaded from xxx. The original raw data can be downloaded from xxx.