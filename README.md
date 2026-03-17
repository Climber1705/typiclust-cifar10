# TypiClust on Cifar10

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Build Representation Model (SimCLR)
```bash
python3 -m scripts.train_simclr
```