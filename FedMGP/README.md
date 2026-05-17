# FedMGP: Personalized Federated Learning with Multi-Group Text-Visual Prompts

Official implementation of **FedMGP: Personalized Federated Learning with Multi-Group Text-Visual Prompts** [NeurIPS 2025]

[[Paper]](https://arxiv.org/abs/2511.00480)

## Abstract

> In this paper, we introduce FedMGP, a new paradigm for personalized federated prompt learning in vision-language models (VLMs). Existing federated prompt learning (FPL) methods often rely on a single, text-only prompt representation, which leads to client-specific overfitting and unstable aggregation under heterogeneous data distributions. Toward this end, FedMGP equips each client with multiple groups of paired textual and visual prompts, enabling the model to capture diverse, fine-grained semantic and instance-level cues. A diversity loss is introduced to drive each prompt group to specialize in distinct and complementary semantic aspects, ensuring that the groups collectively cover a broader range of local characteristics. During communication, FedMGP employs a dynamic prompt aggregation strategy based on similarity-guided probabilistic sampling: each client computes the cosine similarity between its prompt groups and the global prompts from the previous round, then samples groups via a softmax-weighted distribution. This soft selection mechanism preferentially aggregates semantically aligned knowledge while still enabling exploration of underrepresented patterns—effectively balancing the preservation of common knowledge with client-specific features. Notably, FedMGP maintains parameter efficiency by redistributing a fixed prompt capacity across multiple groups, achieving state-of-the-art performance with the lowest communication parameters (5.1k) among all federated prompt learning methods.

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA >= 11.8 (for GPU support)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/FedMGP.git
cd FedMGP
```

### Step 2: Create Conda Environment

```bash
conda create -n fedmgp python=3.10 -y
conda activate fedmgp
```

### Step 3: Install PyTorch

Install PyTorch with CUDA support. Choose the command based on your CUDA version:

```bash
# For CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Dassl Library

The project relies on the [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) library which is included in this repository. Install it as follows:

```bash
cd Dassl
pip install -e .
cd ..
```

> **Note**: If the `Dassl` folder does not contain a `setup.py`, the library is used as a local module and no separate installation is needed.

### Step 5: Install Project Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from Dassl.dassl.config import get_cfg_default; print('Dassl loaded successfully')"
```

---

## Dataset Preparation

Please refer to [DATASETS.md](DATASETS.md) for detailed instructions on downloading and organizing the datasets.

### Supported Datasets

| Type          | Datasets                                                                                                                          |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
| Single-Domain | Caltech101, OxfordPets, OxfordFlowers, DTD, Food101, EuroSAT, UCF101, StanfordCars, FGVCAircraft, SUN397, CIFAR10, CIFAR100, ImageNet |
| Multi-Domain  | DomainNet, Office-Caltech10                                                                                                       |

### Set Dataset Path

Set the environment variable to point to your dataset directory:

```bash
# Add to your ~/.bashrc or ~/.zshrc for persistence
export COOP_DATASET=/path/to/your/datasets

# Or set it temporarily
export COOP_DATASET=$(pwd)/DATA
```

---

## Quick Start

### Training

Run a training experiment on the Caltech101 dataset:

```bash
# Activate environment
conda activate fedmgp

# Set dataset path
export COOP_DATASET=/path/to/your/datasets

# Run training
bash scripts/FedMGP/base2novel_train.sh 0 caltech101 16
```

Parameters:

- `0`: GPU ID
- `caltech101`: Dataset name
- `16`: Number of shots (samples per class)

### Testing

Evaluate the trained model:

```bash
# Test on base classes
bash scripts/FedMGP/base2novel_test.sh 0 caltech101 base 16

# Test on new classes (generalization)
bash scripts/FedMGP/base2novel_test.sh 0 caltech101 new 16
```

---

## Usage

### Running with Scripts

We provide shell scripts for common experiment configurations:

```bash
# Single-domain base-to-new evaluation
bash scripts/FedMGP/base2novel_train.sh <GPU_ID> <DATASET> [SHOTS] [CONFIG]

# Multi-domain training (Office-Caltech10)
bash scripts/FedMGP/office_train.sh <GPU_ID> [SEED]

# Multi-domain training (DomainNet)
bash scripts/FedMGP/domainnet_train.sh <GPU_ID> [SEED]
```

### Running with Python Directly

For more control, run the Python entry point directly:

```bash
CUDA_VISIBLE_DEVICES=0 python federated_main.py \
  --root $COOP_DATASET \
  --model FedMGP \
  --trainer FedMGP \
  --config-file configs/trainers/FedMGP/base2novel_vit_b16.yaml \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --output-dir output/caltech101/FedMGP/test \
  --num_shots 16 \
  --seed 0
```

### Key Command-Line Arguments

| Argument                | Description                                          | Default |
|-------------------------|------------------------------------------------------|---------|
| `--root`                | Path to dataset directory                            | `DATA/` |
| `--model`               | Aggregation model: `FedMGP`, `fedavg`, `local`       | -       |
| `--trainer`             | Trainer name: `FedMGP`, `FedPGP`, `PromptFL`, etc.   | -       |
| `--config-file`         | Path to trainer config YAML                          | -       |
| `--dataset-config-file` | Path to dataset config YAML                          | -       |
| `--output-dir`          | Directory to save outputs                            | -       |
| `--num_shots`           | Number of shots per class                            | 16      |
| `--seed`                | Random seed                                          | 0       |
| `--eval-only`           | Run evaluation only                                  | False   |
| `--subsample`           | Class subset: `all`, `base`, `new`                   | `base`  |

---

## Configuration

### Config File Hierarchy

Configuration is loaded in the following order (later overrides earlier):

1. `configs/datasets/<dataset>.yaml` - Dataset-specific settings
2. `configs/trainers/<trainer>/<config>.yaml` - Algorithm-specific settings
3. Command-line arguments

### Key Configuration Parameters

```yaml
# FedMGP-specific parameters (configs/trainers/FedMGP/base2novel_vit_b16.yaml)
TRAINER:
  FEDMGP:
    N_CTX_VISION: 2          # Vision prompt token length
    N_CTX_TEXT: 2            # Text prompt token length
    NUM_PROMPTS_VISION: 5    # Number of vision prompt groups
    NUM_PROMPTS_TEXT: 5      # Number of text prompt groups
    TOPK: 2                  # Top-k groups for aggregation
    SELECT_MODE: "similarity"  # Selection mode: similarity, random, all
    INFERENCE_MODE: "average"  # Inference mode: average, max_logits
    USE_DIVERGENT_LOSS: True   # Enable diversity loss
    DIVERGENT_LOSS_WEIGHT: 1.0 # Weight of diversity loss

# Federated learning parameters
OPTIM:
  ROUND: 10                  # Number of global communication rounds
  MAX_EPOCH: 2               # Local training epochs per round
  LR: 0.001                  # Learning rate

DATASET:
  USERS: 10                  # Number of federated clients
  NUM_SHOTS: 16              # Shots per class (few-shot setting)
  BETA: 0.3                  # Dirichlet concentration (non-IID degree)
  PARTITION: "noniid-labeldir"  # Data partition strategy
```

---

## Supported Algorithms

| Algorithm  | Model     | Trainer     | Description                           |
|------------|-----------|-------------|---------------------------------------|
| FedMGP     | `FedMGP`  | `FedMGP`    | Multi-group prompt learning (ours)    |
| FedPGP     | `FedPGP`  | `FedPGP`    | Prompt group federated learning       |
| FedOPT     | `FedOPT`  | `FedOPT`    | Optimal transport aggregation         |
| PromptFL   | `fedavg`  | `PromptFL`  | Basic prompt federated learning       |
| IVLP       | `fedavg`  | `IVLP`      | Independent vision-language prompts   |
| VPT        | `fedavg`  | `VPT`       | Visual prompt tuning                  |
| MaPLe      | `fedavg`  | `MaPLe`     | Multi-modal prompt learning           |
| CLIP       | `CLIP`    | `CLIP`      | Zero-shot baseline                    |

---

## Output Structure

```text
output/<dataset>/<trainer>/<config>/<shots>shots/seed<seed>/
|-- logs/              # Training logs
|-- prompt_*.pt        # Saved prompt weights
|-- results.txt        # Evaluation results
|-- best_model.pt      # Best model checkpoint
|-- last_model.pt      # Last epoch checkpoint
```

---

## Citation

If you find our work useful, please cite:

```bibtex
@article{bo2025fedmgp,
  title={FedMGP: Personalized Federated Learning with Multi-Group Text-Visual Prompts},
  author={Bo, Weihao and Sun, Yanpeng and Wang, Yu and Zhang, Xinyu and Li, Zechao},
  journal={arXiv preprint arXiv:2511.00480},
  year={2025}
}
```