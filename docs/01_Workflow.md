# 01_Workflow (low‑RAM, Chromebook‑safe)

## Setup
```
python3 -m venv .venv
./.venv/bin/pip install -U pip scikit-learn scipy
```

## √t band analysis on Zenodo txt
```
python3 prove_transform.py --mode zenodo --file data/zenodo_5790768/Schizophyllum
