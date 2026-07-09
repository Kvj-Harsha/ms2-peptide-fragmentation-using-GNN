# PROJECT_REVIEW.md — Rigorous Technical Review

**Project:** Peptide Fragment Ion Probability Prediction using Graph Neural Networks (PepFragGNN)
**Reviewer role:** Senior ML/proteomics researcher acting as Reviewer #2 for a top venue
**Date:** 2026-07-07
**Verdict (current state):** *Reject — engineering prototype, not yet a research contribution.* Fixable to a workshop/short-paper standard in one week; see `RESEARCH_PLAN.md` and `MASTER_PLAN.md`.

---

## 1. What the project actually is (Phase 1)

### 1.1 Problem it solves
Given a peptide sequence, predict the **empirical probability** that each backbone fragment ion (b- and y-ions) is observed in HCD MS/MS spectra. Probabilities come from the **Pep2Prob** dataset, where each value is the fraction of spectra (for a given precursor) in which a fragment's intensity exceeds a threshold. This is *fragment-presence/probability* prediction, a cousin of *fragment-intensity* prediction (Prosit, pDeep, AlphaPeptDeep).

### 1.2 Core hypothesis (as implemented)
"A GNN over the peptide-as-graph representation can predict position-resolved b/y fragment probabilities from sequence alone." The stated biochemical motivation (message passing captures local + global dependencies) is reasonable in the abstract.

### 1.3 Contributions, honestly assessed
- **Novel scientific contribution:** ~None as it stands. The dataset is not new (Pep2Prob, 2025), the task is not new (fragment prediction is a decade-old field), and the architecture (GCN + mean pool + MLP) is a textbook baseline.
- **Engineering contribution:** A working end-to-end pipeline (data → graph → model → Streamlit demo → optional LLM narration). Competent glue code, but nothing a reviewer would call a system contribution.
- **Maturity level:** **Prototype / MVP.** It runs and produces a number. It is not reproducible against its own README, has no baselines, and its headline metric is not the one the field (or the dataset's own benchmark) uses.

### 1.4 Hidden assumptions
1. That a **single graph-level embedding** (after global mean pooling) can encode *position-specific* fragmentation for up to 39 positions × 2 ion types. This is the central flaw (§2.1).
2. That **charge state is irrelevant** (config hard-codes charge 1 only). Pep2Prob spans charges 1–8 and fragment charges 1–3; fragmentation is strongly charge-dependent (mobile-proton model).
3. That **Pearson correlation** across mixed b/y channels is a meaningful quality metric. It is not the benchmark metric and is easy to inflate (§2.5).
4. That taking the **first N rows** of the dataset is a representative sample. It is not (§2.4).

---

## 2. Critical review — flaws by severity (Phase 2)

Severity scale: **Critical** (invalidates the result), **High** (a reviewer will reject on this), **Medium** (must fix for credibility), **Low** (polish).

### 2.1 [CRITICAL] Global mean pooling destroys the positional signal the task requires
`src/model.py` runs GCN layers, then `global_mean_pool` to a single `[hidden_dim]` vector, then an MLP maps that one vector to all 78 outputs (b1–b39, y1–y39).

- **Why it's fatal:** The output is 78 *position-specific* probabilities, but the model has collapsed the peptide to one order-insensitive summary before predicting them. The mapping "graph embedding → position i's probability" lives entirely in fixed MLP weights that are *identical for every peptide*. The network can only learn a **composition-conditioned average fragmentation curve**; it structurally cannot express "position 3 fragments strongly but position 4 doesn't" in a peptide-specific way beyond whatever leaks through the pooled mean.
- **Failure scenario:** Two peptides that are near-anagrams (similar composition, different order) collapse toward similar pooled embeddings and therefore similar predicted 78-vectors, even though their true fragmentation ladders differ. The model cannot localize cleavage.
- **This contradicts the project's own thesis.** The stated reason to use a GNN is to capture position/structure — but pooling throws exactly that away. A GNN whose output is graph-level is the *wrong tool* for a node/edge-level (cleavage-site-level) label.
- **Fix:** Predict at the **cleavage site**, i.e., the **edge/bond** between residue *i* and *i+1*, from that bond's local node embeddings — not from a pooled graph vector. This is the single change that turns the project from "misapplied GNN" into "correct GNN for the task," and it is also the main novelty hook (see FIORA for small molecules in `LITERATURE_REVIEW.md`). Concretely: for cleavage site *k*, read out from `[h_k, h_{k+1}, h_k⊙h_{k+1}, |h_k−h_{k+1}|, pooled_context, charge]` and predict the b/y probabilities at that site.

### 2.2 [CRITICAL] No baselines — and the field's baselines already beat you
The Pep2Prob benchmark paper (arXiv 2508.21076) reports, on the *same dataset* with **Spectral Angle (SA)**:

| Model | L1 ↓ | MSE ↓ | Spectral Angle ↑ | Accuracy ↑ |
|---|---|---|---|---|
| Global (ion/charge only) | 0.244 | 0.099 | 0.558 | 0.699 |
| Bag-of-Fragment | 0.179 | 0.118 | 0.509 | 0.803 |
| Linear Regression | 0.126 | 0.054 | 0.695 | 0.766 |
| ResNet | 0.069 | 0.021 | 0.818 | 0.871 |
| **Transformer** | **0.056** | **0.017** | **0.845** | **0.953** |

The project reports **none** of these metrics and compares to **nothing**. A GNN paper on this dataset that does not report SA/L1 and does not beat (or at least contextualize against) the **Global** and **Transformer** rows is un-publishable. The most damning baseline is **Global / predict-the-mean-profile**: if the GNN barely beats a per-position dataset average, the graph adds nothing.
- **Fix:** Adopt the benchmark's exact metrics and reproduce at minimum the Global, Bag-of-Fragment, and a plain MLP baseline; position the GNN against the Transformer.

### 2.3 [CRITICAL] Train/eval architecture mismatch — reported numbers are not reproducible
- `train.py` trains `PepFragGNN(hidden_dim=128, num_layers=4)`.
- `eval.py` instantiates `PepFragGNN(hidden_dim=64, num_layers=3)` and calls `load_state_dict(...)` (default `strict=True`).
- `predict.py` / `predict_cli_version.py` use `128/4`.

Loading the committed `128/4` checkpoint with `eval.py`'s `64/3` model **raises a shape-mismatch error** — `eval.py` cannot have produced the README's "Test Loss 0.39 / Pearson 0.82." Those numbers are **stale**, from an earlier configuration, and are presented as if current. This is exactly the kind of inconsistency that sinks reproducibility.
- **Fix:** Single source of truth for hyperparameters (a config/argparse), one checkpoint, one eval script, numbers regenerated from the committed artifacts.

### 2.4 [HIGH] Sampling bias — "first N rows" is not a random sample
`PepDataset` uses `load_dataset(split=...)` then slices the first `num_rows` (150k train, 5k test). If the dataset is ordered (by length, charge, protein, or alphabetically), the training subset is a biased slice and the test subset may not match its distribution. Pep2Prob's *own* split is a leakage-controlled graph-component split; slicing rows can silently break that guarantee and does not give an i.i.d. sample.
- **Fix:** Use the dataset's official train/test folds; if subsampling for compute, sample **uniformly at random with a fixed seed**, and report the subset size as a limitation.

### 2.5 [HIGH] The headline metric (per-peptide Pearson over mixed b/y channels) is misleading
`eval.py` computes Pearson across the flattened valid channels of each peptide, returns `0.0` on failure or `<3` points, and averages (including the 0.0s).
- **Why it's inflated:** b- and y-ion probabilities occupy systematically different ranges and there is a strong, peptide-independent positional shape. A model predicting the *global average shape* already scores high Pearson because it matches this gross bimodal/positional structure — Pearson rewards matching the *pattern*, not the *peptide-specific deviation*. So 0.82 Pearson can coexist with near-zero peptide-specific skill.
- **Secondary bugs:** silently substituting `0.0` for undefined correlations biases the mean; bare `except:` hides real errors; correlating across heterogeneous ion types conflates two distributions.
- **Fix:** Report **Spectral Angle** and **L1/MSE** (dataset-standard), plus Pearson computed **per ion-type separately** and, crucially, **relative to the Global baseline** (skill score). Report the distribution, not just the mean.

### 2.6 [HIGH] Charge state and fragment charge are ignored
`config.py` restricts to charge "1"; the model never sees precursor charge. Fragmentation propensity depends heavily on precursor charge (mobile-proton availability). Pep2Prob provides charge; discarding it caps achievable accuracy and ignores a first-order physical variable.
- **Fix:** Concatenate precursor charge (and target fragment charge, if predicting 2+/3+) into the readout features; ideally predict the full 235-channel space or a principled subset.

### 2.7 [HIGH] The "graph" is a path — the GNN is an over-engineered 1-D sequence model
Edges connect only *i↔i+1*. A GCN on a path graph with mean pooling is, functionally, a low-pass-filtered bag-of-amino-acids — strictly *weaker* than a BiLSTM/Transformer over the sequence (which is what Prosit/AlphaPeptDeep use and what the Pep2Prob Transformer uses). As built, the graph formalism **adds cost without inductive-bias benefit**.
- **Why a reviewer cares:** "Why a GNN?" is the first question and the current design has no good answer. The justification only appears once you move to **edge-level cleavage prediction** and/or **enrich the graph** (e.g., add long-range/side-chain/charge-carrier edges, or physicochemical node features) so the graph structure encodes something a plain sequence model doesn't get for free.
- **Fix:** Either (a) make the edge-level reformulation the contribution and show it beats a matched sequence model, or (b) enrich graph topology (skip edges to basic residues R/K/H, proline-adjacency edges) so message passing captures charge-directed effects.

### 2.8 [MEDIUM] Loss/target semantics: BCELoss on soft probabilistic targets after a raw Sigmoid
Targets are continuous frequencies in [0,1], not binary labels. `nn.BCELoss` on soft targets is a valid cross-entropy between Bernoulli parameters, but:
- Using `nn.BCELoss` on a post-`Sigmoid` output (rather than `BCEWithLogitsLoss`) forgoes the log-sum-exp numerical stabilization.
- BCE is not obviously the right objective for a *regression toward a probability*; the benchmark optimizes/reports L1/MSE/Spectral Angle. Mismatched train objective vs. eval metric is a defensible modeling choice only if justified and ablated.
- **Fix:** Compare BCEWithLogits vs. MSE vs. a spectral-angle / cosine loss; report which trains best for the SA metric. Move `Sigmoid` out of the module and into the loss for stability.

### 2.9 [MEDIUM] Masking with sentinel `-1.0` is fragile
`targets.py` uses `row.get(key_str, -1.0)` and treats `val == -1.0` as "invalid." If any true probability were ever exactly stored as a negative sentinel elsewhere, or if a valid `0.0` is confused with masked, the mask silently corrupts. Float `== -1.0` comparison is brittle.
- **Fix:** Derive the mask from **peptide length** (valid positions = 1..L−1) and explicit key presence, not from a magic float.

### 2.10 [MEDIUM] Wasted capacity / fixed 39-position output
`out_dim=78` assumes ≤40-mers; most tryptic peptides are ~8–15 residues, so the vast majority of the 78 outputs are masked-out for any given peptide. Capacity is spent on positions that never receive gradient for short peptides, and the fixed cap silently truncates 40-mers. The edge-level reformulation (§2.1) removes this problem entirely (predict per existing bond).

### 2.11 [MEDIUM] `build_peptide_graph` crashes / degenerates on edge cases
`pos_norm = i/(L-1)` divides by zero for length-1 peptides; length-1 also yields an empty `edges` list and a malformed `edge_index`. Non-standard residues (the README claims an "unknown token" but the code has none — 20-dim one-hot, so unknown AAs become all-zero vectors, silently). Selenocysteine (U), pyrrolysine (O), and modified residues are unhandled.
- **Fix:** Guard L=1; add an explicit UNK channel (the code claims 21 = 20 + UNK but actually 21 = 20 + position, so the README is wrong — the model has *no* unknown-AA handling).

### 2.12 [MEDIUM] Documentation contradicts code (credibility risk)
- README: "21-dim one-hot (20 AA + unknown)." Code: 20 AA one-hot + 1 position; no unknown channel.
- README §9: "Prediction limited to b1 and y1 ions." Code predicts b/y over 39 positions at charge 1 — "b1/y1" appears to be sloppy shorthand for "b/y charge-1," which will confuse reviewers (b1 = the b-ion at position 1).
- `plan.md` cites a ChatGPT share link and shows different epoch/loss numbers than README.
- Dataset stats disagree (README "610,117"; plan.md "3,148,198 rows"; true = **608,780 precursors** per the source paper).
- **Fix:** One authoritative, internally consistent description regenerated from actual runs.

### 2.13 [LOW] Reproducibility hygiene
No global seed control (data split, shuffling, init all unseeded → non-deterministic results, no CI). No `requirements` pinning of `torch-geometric` CUDA build in a clean file (the current `requirements.txt` is a full conda freeze with local `file:///` paths that won't install on another machine). No experiment logging (no config, no metric history, no `wandb`/CSV). Checkpoints not versioned with their config.

### 2.14 [LOW] Committed clutter / secrets pattern
`src/__pycache__` is committed. `llm_utils.py` hard-codes an API-key slot in source (currently a placeholder — fine now, but the pattern invites a real leak; move to `os.getenv` only). The Streamlit LLM "interpretation" is a nice demo but is **not** a research contribution and should be scoped out of the paper.

---

## 3. Statistical & experimental rigor gaps

- **No significance testing, no confidence intervals, no seeds×runs.** A single number with no variance is not evidence. Pep2Prob reports ± across folds; you must too.
- **No ablations.** Nothing isolates the contribution of: pooling vs. edge readout, charge feature, graph depth, node features, loss function.
- **No error analysis.** No breakdown by peptide length, charge, residue identity, or known biochemical effects (proline effect, Asp/Glu enhancement, basic-residue charge retention). This is where a GNN paper earns its "why graphs" keep.
- **No runtime/memory benchmarking** despite "efficiency on a laptop" being the natural angle (AlphaPeptDeep's whole pitch is 4 M params, 40× faster than Prosit-Transformer — you can compete *there*).
- **Generalization untested:** no length-extrapolation test, no held-out-protein test, no cross-charge test.

---

## 4. Strengths (keep these)

1. **Correct, timely dataset choice.** Pep2Prob is brand-new (2025) and under-explored; being early on it is an opportunity.
2. **Working end-to-end pipeline** including a usable demo — good for a "system + model" workshop paper and for reproducibility optics once fixed.
3. **The task is real and useful.** Fragment-probability prediction genuinely improves peptide-spectrum-match scoring; the motivation section is sound.
4. **The fix is cheap and the novelty is real once reframed** (edge-level GNN for peptide fragmentation is unoccupied ground; FIORA does it for small molecules, not peptides).

---

## 5. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GNN fails to beat Pep2Prob Transformer on SA | High | High | Compete on **efficiency/parameters/interpretability**, not raw SA; target parity at far lower cost (the AlphaPeptDeep angle) |
| Edge-level model still doesn't beat a matched BiLSTM | Medium | High | Include the matched sequence baseline; if graph loses, pivot the story to "when does structure help" analysis |
| One week too short for full 608k training on 6 GB | Medium | Medium | Subsample (random, seeded) to 100–200k; small models (<1 M params) train fast; use gradient accumulation |
| Reviewer says "incremental, dataset already benchmarked" | High | High | Lead with the edge-centric formulation + biochemical interpretability + laptop-efficiency, not "we ran a GNN" |
| Reproducibility questioned | Medium (currently guaranteed) | High | Fix §2.3/§2.13 on Day 1 |

---

## 6. Bottom line

The current repository is a **competent prototype built on a fatal modeling choice** (graph-level pooling for a node/edge-level task), **no baselines**, a **non-reproducible headline number**, and the **wrong evaluation metric** for its own dataset. None of these are hard to fix. Reframed as a **cleavage-site-centric (edge-level) GNN for peptide fragment-ion probability on Pep2Prob**, evaluated with the benchmark's metrics against its published baselines, with biochemical interpretability and laptop-scale efficiency as the selling points, it becomes a credible **workshop / short-paper** contribution within a week. See `RESEARCH_PLAN.md`, `PUBLICATION_ROADMAP.md`, and `MASTER_PLAN.md`.
