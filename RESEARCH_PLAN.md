# RESEARCH_PLAN.md — Methodology, Novelty, and Experimental Design

**Working title:** *CleavageGNN: A Cleavage-Site-Centric Graph Neural Network for Peptide Fragment-Ion Probability Prediction*
**One-line thesis:** Predicting b/y fragment probabilities as an **edge-level (bond/cleavage-site) task** on the peptide graph — rather than a graph-level regression — aligns the model's inductive bias with the physics of backbone dissociation, matching a Transformer baseline on Pep2Prob at a fraction of the parameters while recovering known biochemical fragmentation rules.

---

## 1. Novelty analysis (Phase 4)

**Classification of the *current* project:** Pure engineering / misapplied-method prototype. **Novelty ≈ 2/10.**
**Classification of the *reframed* project:** Moderate research contribution (method + analysis on a new benchmark). **Achievable novelty ≈ 5.5–6.5/10** in one week — enough for a good workshop or a mid-tier conference (IEEE BIBM / ACM-BCB) or a bioinformatics short paper; not enough alone for NeurIPS/ICLR main track.

**Why the reframe creates real novelty (see `LITERATURE_REVIEW.md` §F):**
1. **First GNN baseline on Pep2Prob** — the benchmark has none (its "graph" is only for the data split).
2. **First cleavage-site/edge-centric formulation for peptide fragment probability** — FIORA proved the bond-local principle for *small molecules*; the peptide-backbone analogue on probability data is unoccupied.
3. **Biochemical interpretability** — testing whether a learned model recovers the mobile-proton and proline effects is novel *on this dataset* and cheap.
4. **Parameter/energy efficiency** — a <1 M-param GNN trainable on a 6 GB laptop, contextualized against the 4 M / 64 M sequence-model lineage (AlphaPeptDeep/Prosit).

**Ways to strengthen novelty, ranked by novelty-to-effort (do 1–3; 4–5 if time):**
| # | Idea | Type | Effort | Novelty gain |
|---|---|---|---|---|
| 1 | Edge-level readout at each bond `k` from `[h_k,h_{k+1},h_k⊙h_{k+1},|h_k−h_{k+1}|,ctx,charge]` | Architectural | Low | High |
| 2 | Charge / mobile-proton features + charge-carrier edges (skip-edges to R/K/H) | Architectural + domain | Low | High |
| 3 | Interpretability: attention/gradient attribution vs known motifs (Pro, Asp) | Analysis | Low | Medium-High |
| 4 | Spectral-angle-aware loss vs BCE vs MSE ablation | Objective | Low | Medium |
| 5 | Multi-charge fragment extension (charge 1–2 → full 235-channel subset) | Task scope | Medium | Medium |

---

## 2. Proposed methodology

### 2.1 Task formalization
For a peptide of length `L` with residues `r_1…r_L` and precursor charge `z`, predict for each **cleavage site** `k ∈ {1,…,L−1}` the probabilities of the resulting b- and y-ions at fragment charge(s) `c`:
`ŷ_{k} = f_θ(graph, z)_k ∈ [0,1]^{|ions|×|charges|}`.
Train on Pep2Prob's empirical probabilities; **use the official leakage-controlled folds and metrics.**

### 2.2 Graph construction (fix + enrich)
- **Nodes:** residues. **Node features (replace one-hot-only):** 20-dim AA one-hot **+ explicit UNK channel**, + physicochemical scalars (basicity/proton affinity, hydrophobicity, mass, pKa of side chain), + normalized position (guard `L=1`), + is-basic (R/K/H) flag, + is-Pro / is-Asp/Glu flags.
- **Edges (enrich beyond the path):** (a) backbone `i↔i+1`; (b) optional **charge-carrier skip edges** connecting basic residues to all backbone bonds (mobile-proton reach); (c) **N-/C-terminal virtual nodes**. Ablate (b)/(c) to show the graph earns its keep.
- **Global context:** precursor charge `z`, peptide length, total basic-residue count → injected at readout and/or as a virtual global node.

### 2.3 Model (`CleavageGNN`)
- Backbone: `K` message-passing layers (compare **GCN → GAT → GIN**; expect GIN/GAT > GCN per `LITERATURE_REVIEW.md` D3/D4). Residual connections + LayerNorm; hidden 64–128.
- **Edge readout head** (the core change): for bond `k`, `MLP([h_k, h_{k+1}, h_k⊙h_{k+1}, |h_k−h_{k+1}|, g_context, emb(z)])`. Output logits → per-fragment probabilities.
- Keep total params **< 1 M** (headline efficiency claim). Move `Sigmoid` into the loss (`BCEWithLogits`) for stability.

### 2.4 Training
- Loss: default `BCEWithLogitsLoss` on soft targets with length-derived mask; **ablate** vs MSE and a spectral-angle loss.
- Optimizer AdamW, cosine schedule, early stopping on val Spectral Angle. Fixed seeds (≥3 seeds for the headline model).
- Mask from **peptide length + key presence**, never a `-1.0` sentinel.

---

## 3. Experimental design (Phase 5 — all feasible in <1 week on RTX 4050, 6 GB)

### 3.1 Datasets / splits
- Pep2Prob official folds. For compute, **seeded uniform-random subsample** (start 100k train / 20k val / 20k test; scale to full if time). Report subset size as a limitation and show a scaling curve.

### 3.2 Baselines (reproduce or re-report)
1. **Global** (per-position dataset mean) — the "does the graph do anything" floor.
2. **Bag-of-Fragment / Bag-of-AA MLP.**
3. **Matched sequence model** — a small **BiLSTM** *and/or* re-run of the Pep2Prob **Transformer** at comparable parameter budget (the key head-to-head).
4. **Ablated GNN**: graph-level mean-pool version (the *original* model) vs edge-level (shows the reformulation's value).

### 3.3 Metrics (adopt the benchmark's)
- Primary: **Spectral Angle (SA)**, **L1**, **MSE** (type-1). Report ± across folds/seeds.
- Secondary: Accuracy/Sensitivity/Specificity at τ=0.001 (type-2).
- Diagnostic: per-ion-type Pearson **as a skill score vs Global** (never the raw inflated number from the old `eval.py`).

### 3.4 Ablations
- Readout: **edge-level vs global-mean-pool** *(headline ablation)*.
- Backbone: GCN vs GAT vs GIN; depth `K∈{2,3,4,5}`.
- Features: one-hot only vs +physicochemical vs +charge.
- Edges: path-only vs +charge-carrier skip edges vs +virtual global node.
- Loss: BCE vs MSE vs spectral-angle.

### 3.5 Robustness & generalization
- **Length extrapolation:** train on ≤20-mers, test on >20-mers.
- **Charge generalization:** hold out a precursor charge, test on it.
- **Held-out peptides** already guaranteed by the leakage-controlled split — state it.

### 3.6 Error / qualitative analysis (the "why graphs / science" section)
- Stratify SA by length, charge, basic-residue count.
- **Motif recovery:** average predicted vs true enhancement N-terminal to Pro, C-terminal to Asp/Glu; charge retention at R/K/H. Figure: predicted cleavage propensity aligned to the mobile-proton model (`LITERATURE_REVIEW.md` E1/E2).
- **Attribution:** GAT attention or integrated gradients on bonds → do high-probability sites coincide with known chemistry?

### 3.7 Systems benchmarks
- Params, train time/epoch, peak VRAM, inference throughput (peptides/s) on the 4050 — vs the Transformer baseline. This table is a selling point.

### 3.8 Statistical rigor
- ≥3 seeds for headline models; report mean ± std. **Paired significance test** (paired t-test or Wilcoxon signed-rank across per-peptide SA) for edge-level vs pooled and vs BiLSTM. Bootstrap 95% CIs on SA.

---

## 4. Success criteria
- **Must-have (paper viable):** edge-level GNN **beats** the global-pool version and the Global baseline on SA, with significance; reproducible numbers from committed code; ≥1 clean interpretability figure.
- **Strong result:** GNN **matches the Pep2Prob Transformer SA (~0.845)** at **<25% of its parameters** and trains on a 6 GB laptop.
- **Fallback story (if GNN < Transformer on SA):** "Structure-aware but cheaper + interpretable" — lead with efficiency + biochemical recovery + the *when-does-structure-help* analysis; still workshop-worthy.

---

## 5. Proposed contributions (for the paper's intro bullet list)
1. The **first GNN** for peptide fragment-ion **probability** on Pep2Prob, formulated as an **edge-level cleavage-site** prediction that matches the physics of backbone dissociation.
2. A **charge/mobile-proton-aware** graph design (charge-carrier edges, physicochemical features) and a controlled ablation isolating each component.
3. **Biochemical interpretability**: evidence the learned model recovers mobile-proton, proline, and acidic-residue effects.
4. A **parameter- and compute-efficient** model (<1 M params, trainable on a 6 GB laptop) that is competitive with a Transformer baseline, with full runtime/memory accounting.
5. **Reproducible** code, configs, seeds, and the exact benchmark metrics/splits.
