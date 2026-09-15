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

The arguments are `DATASET`, `ALPHA`, `GPU_ID`, optional `ABLATION_MODE`
(default `full`), and optional `SEED` (default `7`). The paper settings
use 20 clients, 8 online clients per round, 5 local epochs, seed 7,
`lambda_u=1.0`, `lambda_kl=0.5`, local/global prior smoothing of `1.0/0.1`,
and correction strength `0.25`. Dataset-specific communication rounds and
labeled samples per class are set by `FedGPLA.py`:

| Dataset | Rounds |
| --- | ---: | 
| CIFAR-10 | 300 | 
| CIFAR-100 | 500 |
| SVHN | 150 |
| CINIC-10 | 400 |

To launch the complete 12-setting grid sequentially, run:

```bash
for dataset in CIFAR10 CIFAR100 SVHN CINIC10; do
  for alpha in 0.1 0.5 1.0; do
    bash scripts/run.sh "$dataset" "$alpha" 0
  done
done
```

Metrics and logs are written to a unique run directory under
`results/<mode>/<dataset>/alpha_<alpha>/seed_<seed>/run_<timestamp>_<suffix>/`.
Each launch creates its own directory, including simultaneous runs with the
same mode and seed, so previous outputs are never overwritten.
Add `--save_checkpoints` when calling `FedGPLA.py` directly if model
checkpoints are also required.

All command-line options can be inspected with:

```bash
python FedGPLA.py --help
```

## Prior-construction ablations

Select one of five modes with `--ablation_mode` or the fourth argument to
`scripts/run.sh`:

| Mode | Unlabeled class statistics | Global reference for logit adjustment |
| --- | --- | --- |
| `full` | Reliable predictions contribute one-hot counts; other predictions contribute soft probabilities | Smoothed sum of the latest statistics of all clients, retained in client-wise memory |
| `all_hard` | Every prediction contributes a one-hot count from the argmax of the eight-view average | Same memory mechanism as `full`, using the modified statistics |
| `all_soft` | Every prediction contributes the eight-view average probability vector | Same memory mechanism as `full`, using the modified statistics |
| `uniform` | Same hybrid counts as `full` | Fixed to `1 / num_classes` |
| `without_memory` | Same hybrid counts as `full` | Smoothed sum of only the current round's participating-client statistics |

All modes retain the eight weak views and the strong view in evidence
estimation. In `full`, reliability requires both confidence at least `0.95`
and agreement of the weak and strong predicted classes. Labeled counts
remain unchanged, and every unlabeled example contributes total mass one.
The same local evidence vector constructs the local prior and is uploaded
to the server. These changes do **not** harden the soft pseudo-labels used
in the training loss; GGCR and pseudo-label confidence filtering are also
unchanged.

`without_memory` starts semi-supervised training with exactly the same
smoothed all-client labeled-count prior as `full`. At each subsequent
round end, it replaces the reference using only that round's uploads,
for use in the **next** round. It retains no per-client history and does
not repeat initialization. `uniform` uses a uniform reference throughout
the semi-supervised stage, including its first round.

Run one mode, or run all five sequentially on the same GPU:

```bash
bash scripts/run.sh CIFAR10 0.5 0 all_hard 7
bash scripts/run_ablations.sh CIFAR10 0.5 0 7
bash scripts/run_ablations.sh CIFAR100 0.5 0 7
```

The launcher fixes correction strength to `0.25`, GGCR weight to `0.5`,
and all other paper settings equally across modes. The original direct
CLI hyperparameter options remain available for separate sensitivity
experiments. For a controlled mode comparison, use the same settings,
seed and software environment in every run. Mode selection does not
consume random draws; the existing data split and dedicated client
sampling generator are unchanged. In particular, the original
Dirichlet helper still receives `seed=0`, while the experiment seed
controls labeled-sample selection and client participation.

Each run directory contains:

- `accuracy.csv`: global test accuracy per round, in the original scale.
- `metrics.csv`: accuracy, pseudo-label accuracy over all evaluated
  pseudo-labels, selected fraction/count, and the global GeM parameter.
- `logs/train.log`: training losses and diagnostics, including the mode.
- `config.json`: resolved arguments, warm-up length and evidence-view count.
- `partition.json`: actual labeled and truly unlabeled indices per client,
  before labeled examples are added to the unlabeled training pool.
- `clients.jsonl`: participating clients in their training order each round.
- `checkpoints/`: model weights when checkpoint saving is enabled.

Compare `partition.json` and `clients.jsonl` between modes to verify the
paired experiment setup. These diagnostics do not use unlabeled ground-truth
labels for training or prior construction.

Behavioral checks for evidence construction, global prior updates and run
isolation run on CPU without downloading datasets:

```bash
python -m unittest discover -s tests -v
```

## Acknowledgements

This implementation builds on the public
[ProxyFL](https://github.com/DuowenC/FSSLlib) codebase (local base commit
`f732a27`) and its acknowledged upstream project
[SAGE](https://github.com/Jay-Codeman/SAGE). The RandAugment implementation
retains its upstream attribution comments in `Dataset/randaugment.py`.
