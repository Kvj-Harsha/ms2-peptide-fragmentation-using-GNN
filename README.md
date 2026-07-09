# CleavageGNN — Peptide Fragment-Ion Probability Prediction with Graph Neural Networks

A cleavage-site-centric graph neural network for predicting **b/y fragment-ion
probabilities** on the [Pep2Prob](https://huggingface.co/datasets/bandeiralab/Pep2Prob)
benchmark (608,780 precursors aggregated from ~183M spectra).

Rather than pooling a peptide into a single vector and predicting every position
from shared MLP weights, **CleavageGNN predicts each fragment at its cleavage-site
edge** — aligning the model's inductive bias with the physics of backbone
dissociation. See `RESEARCH_PLAN.md` and `PROJECT_REVIEW.md` for the scientific
motivation and the rationale behind this reformulation.

---

## Why edge-level?

For a peptide of length `L` and precursor charge `z`, each cleavage site
`k ∈ {1..L-1}` produces a b-ion (`b_k`) and a y-ion (`y_{L-k}`). CleavageGNN reads
each site out from its own bond:

```
y_k = MLP([ h_k, h_{k+1}, h_k ⊙ h_{k+1}, |h_k − h_{k+1}|, ctx, emb(z) ])
```

where `h_k` are GNN node embeddings, `ctx` is a pooled global context, and
`emb(z)` is the precursor-charge embedding. The pooled global-mean model is kept
as the **headline ablation** (`--model.readout pool`) to isolate the value of the
edge formulation.

---

## Project structure

```
pepfraggnn/                 installable package (single source of truth)
├── config.py               all hyperparameters (dataclass + argparse/YAML)
├── seed.py                 reproducibility (seeds, device)
├── metrics.py              Spectral Angle, L1, MSE, type-2 accuracy
├── losses.py               masked BCEWithLogits / MSE / spectral-angle
├── engine.py               shared train / eval loops (edge & pool readouts)
├── utils.py                param counting, CSV logging
├── data/
│   ├── features.py         AA vocab (20 + explicit UNK) + physicochemistry
│   ├── graph.py            peptide → graph (L=1 guard, charge-carrier edges)
│   ├── targets.py          length-derived masking (no −1.0 sentinel)
│   ├── loader.py           seeded uniform-random subsampling / official folds
│   └── dataset.py          PyG dataset (edge + pooled targets, charge)
├── models/                 GNN backbones (GCN/GAT/GIN), CleavageGNN, PooledGNN
└── baselines/              Global mean profile, Bag-of-AA MLP
scripts/                    train.py, evaluate.py, run_baselines.py, predict.py, reproduce.py
configs/                    YAML run configs
app/                        Streamlit demo (+ optional LLM narration)
tests/                      pytest unit + integration tests
```

---

## Setup (virtual environment)

```bash
python -m venv .venv
# Windows PowerShell:   .venv\Scripts\Activate.ps1
# Git Bash / macOS/Linux: source .venv/bin/activate

# PyTorch: pick the build for your machine
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu121 # CUDA 12.1

pip install -r requirements.txt
pip install -e .          # optional: install pepfraggnn as a package
```

---

## Usage

**Fully offline smoke test (no download) — runs the model end-to-end on synthetic data:**
```bash
pytest -q
```

**Smoke-test the real pipeline** (downloads Pep2Prob on first run, then subsamples to a few hundred rows):
```bash
python scripts/reproduce.py --quick
```

**Reproduce baselines** (Global mean profile + Bag-of-AA MLP):
```bash
python scripts/run_baselines.py --config configs/cleavage_gcn.yaml
```

**Train the edge-level model / pooled ablation:**
```bash
python scripts/train.py --config configs/cleavage_gcn.yaml --model.readout edge --out_dir runs/edge_gcn
python scripts/train.py --config configs/cleavage_gcn.yaml --model.readout pool --out_dir runs/pool_gcn
```

**Evaluate a checkpoint** (writes `results/main_results.csv`):
```bash
python scripts/evaluate.py --ckpt runs/edge_gcn/model.pt
```

**Predict for a single peptide:**
```bash
python scripts/predict.py --ckpt runs/edge_gcn/model.pt --peptide PEPTIDER
```

**Interactive demo:**
```bash
streamlit run app/streamlit_app.py -- --ckpt runs/edge_gcn/model.pt
```

Any config field is overridable on the CLI with dotted flags, e.g.
`--model.backbone gat --train.epochs 30 --train.loss spectral_angle`.

---

## Evaluation

Metrics follow the Pep2Prob benchmark so results are comparable to its published
baselines:

- **Spectral Angle (SA)** — headline metric, reported with per-peptide std.
- **L1** and **MSE** — regression error over valid fragments.
- **Type-2** accuracy / sensitivity / specificity at τ=0.001.

The old inflated per-peptide Pearson is deliberately **not** used as a headline
(see `PROJECT_REVIEW.md` §2.5).

---

## Reproducibility notes

- **One source of truth** for hyperparameters (`pepfraggnn/config.py`); every
  checkpoint stores its own config, so train/eval architecture can never diverge.
- **Fixed seeds** across Python / NumPy / Torch; subsampling is seeded and
  uniform-random (not a biased first-N slice).
- **Node features:** 20 canonical amino acids **+ an explicit UNK channel**, plus
  physicochemical scalars, normalized position, and terminal flags.
- Model stays **< 1M parameters** (asserted in the tests) for laptop-scale training.

Run the tests:
```bash
pytest -q
```

---

## Scope note

The Streamlit demo and optional LLM narration are a **usability bonus**, not a
research claim. The LLM key is read only from `GROQ_API_KEY` in the environment
and is never stored in source.
