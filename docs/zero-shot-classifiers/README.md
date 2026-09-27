# Zero-shot classifier research

Use this directory for model, data, evaluation, and training research on classifiers that accept labels at request time. This scope is wider than the MiniCPM5 pilot or any one GPU provider. Keep the existing five-level command-risk classifier work separate.

## Runpod pilot: local and remote setup

- Local: SSH client, `rsync`, a Runpod SSH public key, and a private key readable only by its owner. Use the SSH host, port, and user shown by Runpod. An SSH alias makes the existing transfer scripts easier to use. The Runpod API key is only needed for API or CLI control, not SSH access to a Pod.
- Pod: SSH access, a working NVIDIA driver and PyTorch CUDA environment, Python 3.13 and `uv`. Mount persistent storage at `/workspace`; keep data, Hugging Face cache, checkpoints, and logs there. Check GPU availability in the chosen location before creating a location-bound volume.
- Transfer: `scripts/sync_to_server.sh` sends this repo to a dedicated remote code directory, but excludes `runs/`, `artifacts/`, generated data, and `.git`. It uses `rsync --delete`, so never point it at the volume root. Stage external datasets separately.
- Recovery: `scripts/fetch_artifacts.sh` only fetches the existing classifier's `best/`, `reports/`, and `run_metadata.json`. Copy all new adapters, resumable checkpoints, configs, logs, and evaluation results back before deleting a Pod; verify the copies. A retained network volume continues to cost money.

The current `model-gym train` command trains ModernBERT for fixed command-risk labels. MiniCPM5 zero-shot/LoRA training is not implemented here yet; it needs a separate data contract, dependencies, training and evaluation path, and a short GPU smoke run before a paid long run.
