# Dataset

This project uses the **New Plant Diseases Dataset (Augmented)** from Kaggle:
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

It's not committed to this repo (too large for git). Download it yourself:

## Option A — Kaggle CLI (recommended, works locally or on Colab)

```bash
pip install kaggle
# Place your kaggle.json API token at ~/.kaggle/kaggle.json (Kaggle account -> Create New Token)
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data --unzip
```

## Option B — Manual download

1. Download the zip from the Kaggle page above.
2. Unzip it into `data/` so the structure looks like:

```
data/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── ... (38 class folders total)
├── valid/
│   ├── Apple___Apple_scab/
│   ├── ... (same 38 class folders)
└── test/
    └── (33 unlabeled images, class inferable from filename)
```

Note: the raw Kaggle zip nests everything one level deeper (e.g.
`New Plant Diseases Dataset(Augmented)/train/...`) — flatten it so `train/` and `valid/`
sit directly under `data/`, or pass `--data-dir` pointing at the nested folder instead.
