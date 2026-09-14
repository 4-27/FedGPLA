# FedGPLA

PyTorch code for reproducing the main FedGPLA results in federated
semi-supervised learning. The release intentionally contains the training
entry point and its direct model/data dependencies.

## Environment

The experiments were developed with Python 3.8.18, PyTorch 2.1.1, and CUDA.

```bash
conda create -n fedgpla python=3.8.18 -y
conda activate fedgpla
pip install -r requirements.txt
```

## Datasets

CIFAR-10, CIFAR-100, and SVHN are downloaded automatically into `data/`.
For CINIC-10, download the dataset separately and arrange it as follows:

```text
data/CINIC10/
├── train/
│   └── <class folders>/
├── valid/
│   └── <class folders>/
└── test/
    └── <class folders>/
```

Use `--path_cinic10 /path/to/CINIC10` if it is stored elsewhere.

## Reproducing the main settings

Run one dataset/Dirichlet-alpha setting on one GPU:

```bash
bash scripts/run.sh CIFAR10 0.1 0
```

The three arguments are `DATASET`, `ALPHA`, and `GPU_ID`. The paper settings
use 20 clients, 8 online clients per round, 5 local epochs, seed 7,
`lambda_u=1.0`, `lambda_kl=0.5`, local/global prior smoothing of `1.0/0.1`,
and correction strength `0.25`. Dataset-specific communication rounds and
labeled samples per class are set by `FedGPLA.py`:

| Dataset | Rounds |
| --- | ---: | 
| CIFAR-10 | 300 | 
| CIFAR-100 | 500 |
| SVHN | 150 | 460 |
| CINIC-10 | 400 |

To launch the complete 12-setting grid sequentially, run:

```bash
for dataset in CIFAR10 CIFAR100 SVHN CINIC10; do
  for alpha in 0.1 0.5 1.0; do
    bash scripts/run.sh "$dataset" "$alpha" 0
  done
done
```

Metrics and logs are written under
`results/<dataset>/alpha_<alpha>/seed_<seed>/`. Add `--save_checkpoints` when
calling `FedGPLA.py` directly if model checkpoints are also required.

All command-line options can be inspected with:

```bash
python FedGPLA.py --help
```

## Acknowledgements

This implementation builds on the public
[ProxyFL](https://github.com/DuowenC/FSSLlib) codebase (local base commit
`f732a27`) and its acknowledged upstream project
[SAGE](https://github.com/Jay-Codeman/SAGE). The RandAugment implementation
retains its upstream attribution comments in `Dataset/randaugment.py`.
