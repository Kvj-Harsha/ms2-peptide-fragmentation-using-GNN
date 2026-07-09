# MASTER_PLAN.md — One-Week Execution Roadmap (RTX 4050, 6 GB)

**Goal:** Turn the current PepFragGNN prototype into a submission-ready **workshop/short-paper** contribution in **7 days** on a single RTX 4050 laptop GPU.
**Read alongside:** `PROJECT_REVIEW.md` (what's broken), `LITERATURE_REVIEW.md` (grounding), `RESEARCH_PLAN.md` (method/experiments), `PUBLICATION_ROADMAP.md` (where to submit).

**The pivot in one sentence:** Stop pooling the peptide into one vector and predicting 78 positions from it; instead predict each fragment at its **cleavage-site edge**, add **charge/mobile-proton** features, evaluate with **Pep2Prob's own metrics against its published baselines**, and sell **interpretability + laptop-scale efficiency**.

---

## 0. Guiding priorities (do these regardless of time)
1. **Correctness before cleverness:** fix the reproducibility bugs (Day 1) before any new modeling.
2. **Adopt the benchmark's metrics/splits (SA, L1, MSE; official folds).** Never report the old inflated Pearson as the headline.
3. **Always beat the `Global`/mean-profile baseline with significance**, or you have no paper.
4. **Keep the model <1 M params** so it trains fast on 6 GB and supports the efficiency story.
5. **Write as you go.** Methods/Experiments drafted Days 2–6.

---

## 1. Prioritized task list (highest impact first)
- **P0 (blocking):** unify train/eval config (fix §2.3 mismatch); seeds; official splits or seeded random subsample; implement SA/L1/MSE eval; reproduce Global + Bag-of-AA baselines.
- **P1 (core novelty):** edge-level cleavage readout; charge feature; length-derived masking; BCEWithLogits.
- **P1 (headline ablation):** global-pool vs edge-level.
- **P2:** GAT/GIN backbones; charge-carrier edges; physicochemical features; loss ablation.
- **P2:** matched BiLSTM (and, if time, re-run Pep2Prob Transformer) for head-to-head.
- **P3:** interpretability (proline/Asp recovery, attention/IG); robustness (length/charge extrapolation); efficiency table; significance tests.
- **P3:** paper writing, figures, reproducibility packaging.

---

## 2. Day-by-day schedule

### Day 1 — Foundation, correctness, baselines (8–9 h)
- **Objectives:** Make the repo reproducible and instrument the *right* evaluation.
- **Tasks:**
  - Add `config.py`/argparse single source of truth; fix the `train.py`(128/4) vs `eval.py`(64/3) mismatch; set global seeds.
  - Replace metric code with **Spectral Angle, L1, MSE** (+ type-2 acc/sens/spec at τ=0.001); compute per-fold ± ; add per-ion Pearson as a *skill score vs Global* only.
  - Switch to **official Pep2Prob folds**; if subsampling, **seeded uniform random** (100k/20k/20k), not first-N.
  - Implement baselines: **Global (per-position mean)**, **Bag-of-AA MLP**. Reproduce their SA/L1 to sanity-check against the paper's Global row (SA≈0.558).
  - Fix `build_peptide_graph` (L=1 guard, explicit UNK channel); fix `-1.0` masking → length-derived mask.
- **Outputs:** Reproducible pipeline; baseline numbers on your subset; a metrics module.
- **Compute:** Low (CPU-bound baselines + small runs). VRAM < 2 GB.
- **Risks:** HF dataset download size/time → start the download first thing; cache locally.
- **Deliverable:** `metrics.py`, `baselines.py`, reproducible `train.py/eval.py`, baseline results CSV.

### Day 2 — Core reformulation: edge-level CleavageGNN (8–9 h)
- **Objectives:** Implement and validate the edge-level readout + charge conditioning.
- **Tasks:**
  - New model: GNN backbone → **edge readout** `MLP([h_k,h_{k+1},h_k⊙h_{k+1},|h_k−h_{k+1}|,ctx,emb(z)])`; `BCEWithLogits`.
  - Add precursor-charge embedding + basic-residue/Pro/Asp flags to features.
  - Train on the 100k subset; confirm it **beats Global** on SA.
  - Keep the **old global-pool model** runnable (it's the headline ablation).
- **Outputs:** Working CleavageGNN; first SA vs Global and vs old-pool numbers.
- **Compute:** Med. Small GNN, batch 32–64, < 4 GB VRAM; ~minutes/epoch on subset.
- **Risks:** PyG edge-batching bugs → unit-test on a 3-peptide batch.
- **Deliverable:** `model_cleavage.py`, first comparison table (edge vs pool vs Global).

### Day 3 — Architecture & feature sweeps (8 h)
- **Objectives:** Pick the best backbone/features; run headline ablation cleanly.
- **Tasks:**
  - Backbone sweep: **GCN vs GAT vs GIN**, depth `{2,3,4,5}`, hidden `{64,128}`.
  - Feature ablation: one-hot vs +physicochemical vs +charge.
  - Edge ablation: path-only vs +charge-carrier skip edges vs +virtual global node.
  - Lock the headline config; run **3 seeds**.
- **Outputs:** Ablation tables; selected model with mean±std.
- **Compute:** Med-High (many short runs). Use the 100k subset; queue runs.
- **Risks:** Sweep sprawl → cap each run's epochs; use early stopping; log to CSV.
- **Deliverable:** `ablations.csv`, chosen config.

### Day 4 — Baselines head-to-head + scale-up (8–9 h)
- **Objectives:** Fair comparison to sequence models; scale toward full data if feasible.
- **Tasks:**
  - Train a **matched BiLSTM**; if time/VRAM allow, re-run the **Pep2Prob Transformer** at comparable params (else re-report their published numbers, clearly labeled).
  - Scale headline GNN to a larger subset (200k–full) with gradient accumulation; produce a **data-scaling curve**.
  - Loss ablation: BCE vs MSE vs spectral-angle loss.
- **Outputs:** Main results table (Global/BoF/BiLSTM/[Transformer]/CleavageGNN); scaling curve.
- **Compute:** High. Watch 6 GB ceiling — reduce batch, use AMP (`torch.cuda.amp`), gradient accumulation.
- **Risks:** OOM on Transformer/large batch → AMP + smaller batch + accumulation; if Transformer won't fit, re-report published SA and note it.
- **Deliverable:** `main_results.csv`, scaling figure.

### Day 5 — Interpretability & robustness (8 h)
- **Objectives:** The "why graphs / science" evidence and generalization tests.
- **Tasks:**
  - **Motif recovery:** predicted vs true enhancement N-term to Pro, C-term to Asp/Glu; charge retention at R/K/H. Aggregate across test set.
  - **Attribution:** GAT attention or integrated gradients on bonds; example-peptide heatmap.
  - Robustness: **length extrapolation** (≤20 → >20), **charge holdout**.
  - Error stratification by length/charge/basicity.
- **Outputs:** Interpretability figures (F3/F4), robustness table.
- **Compute:** Low-Med (mostly inference).
- **Risks:** Weak/negative interpretability → still report honestly; it becomes a "learned vs known chemistry" discussion.
- **Deliverable:** analysis figures + `robustness.csv`.

### Day 6 — Consolidation, stats, packaging (8 h)
- **Objectives:** Final numbers, significance, reproducibility, figures.
- **Tasks:**
  - **Significance:** paired Wilcoxon/t-test (edge vs pool; GNN vs BiLSTM) on per-peptide SA; bootstrap 95% CIs.
  - **Efficiency table:** params, VRAM, train time/epoch, inference throughput on 4050 vs baselines.
  - Freeze results; finalize all figures/tables; write repro README + one-command script; pin a clean `requirements.txt` (drop the conda `file:///` mess); remove committed `__pycache__`.
- **Outputs:** Final tables T1–T3, all figures, frozen checkpoints + configs.
- **Compute:** Low.
- **Deliverable:** `results/` locked, reproducible repo.

### Day 7 — Writing (8–9 h)
- **Objectives:** Complete the paper draft.
- **Tasks:** Fill the outline in `PUBLICATION_ROADMAP.md` §3 (Methods/Experiments already drafted Days 2–6). Write Intro, Related Work, Analysis, Abstract, Limitations, Reproducibility. Self-review against `PROJECT_REVIEW.md` weakness list. Prepare submission (workshop/BIBM formatting).
- **Compute:** None.
- **Deliverable:** Submission-ready draft + arxiv-able repo.

---

## 3. Timeline & milestones
| Milestone | By end of | Gate to pass |
|---|---|---|
| M1: Reproducible pipeline + baselines + correct metrics | Day 1 | Global SA reproduced (~0.55), no config mismatch |
| M2: Edge-level model beats Global | Day 2 | SA(edge) > SA(Global), sanity-checked |
| M3: Headline ablation (edge > pool) with 3 seeds | Day 3 | edge > pool, mean±std |
| M4: Full baseline table + scaling curve | Day 4 | GNN vs BiLSTM/Transformer numbers in hand |
| M5: Interpretability + robustness evidence | Day 5 | ≥1 clean motif figure |
| M6: Stats + efficiency + frozen repro | Day 6 | significance p<0.05 on key claim |
| M7: Paper draft complete | Day 7 | outline fully written, self-reviewed |

---

## 4. Compute budget (RTX 4050, 6 GB VRAM)
- **Model:** <1 M params → training footprint dominated by activations; batch 32–64 fits comfortably.
- **Use:** mixed precision (`torch.cuda.amp`), gradient accumulation for effective large batches, `num_workers>0` dataloading, cache the HF dataset to disk once.
- **Subset strategy:** 100k train for dev (minutes/epoch); scale to 200k–full only for the final headline run and scaling curve.
- **Avoid OOM:** if the Transformer baseline won't fit, re-report Pep2Prob's published numbers (clearly labeled) rather than forcing it.
- **Disk:** Pep2Prob is large (183M spectra aggregated to 608k precursors) — ensure tens of GB free for the HF cache.

---

## 5. Risk register & mitigation
| Risk | Mitigation |
|---|---|
| GNN can't beat Transformer SA | Pivot story to efficiency + interpretability + "when structure helps"; still workshop-worthy |
| One week too tight for full data | Subsample (seeded); report scaling curve; full-data run becomes v2 |
| 6 GB OOM | AMP, small batch + accumulation, <1M params, re-report heavy baselines |
| HF download/cache time | Start Day 1 hour 1; cache locally; work on baselines while it downloads |
| Interpretability inconclusive | Report honestly as an analysis, not a claim |
| Reviewer "incremental" | Foreground edge-reformulation + first-GNN-on-Pep2Prob + biochemistry; two-stage pub plan |
| Deadlines shifted | **Verify all venue deadlines before committing** (see `PUBLICATION_ROADMAP.md`) |

---

## 6. Final checklist for a submission-ready paper
**Reproducibility**
- [ ] Single config source; fixed seeds; one-command repro; clean pinned `requirements.txt`; `__pycache__` removed.
- [ ] Official Pep2Prob splits (or clearly-stated seeded subsample); leakage statement.
- [ ] Checkpoints + configs + result CSVs committed.

**Rigor**
- [ ] SA/L1/MSE (+type-2) reported with ± across ≥3 seeds/folds.
- [ ] Baselines: Global, Bag-of-AA, BiLSTM, (Transformer reproduced or clearly re-reported).
- [ ] Headline ablation (edge vs pool) with **paired significance test** + CIs.
- [ ] Feature/edge/backbone/loss ablations.
- [ ] Robustness: length + charge extrapolation.

**Contribution & story**
- [ ] Edge-level cleavage formulation clearly motivated by mobile-proton chemistry.
- [ ] Interpretability figure (proline/Asp recovery or attribution).
- [ ] Efficiency table (params/VRAM/throughput vs baselines).
- [ ] Contributions bullets match delivered results (no overclaiming).

**Scholarship**
- [ ] Pep2Prob cited and compared directly.
- [ ] Full citation set (`LITERATURE_REVIEW.md` §G), every paper actually read.
- [ ] Limitations mirror the honest constraints (subset, charge scope, PTMs, HCD-only).

**Docs**
- [ ] README rewritten to match code (no "21=20+unknown" error, no "b1/y1" ambiguity, correct dataset stats: 608,780 precursors).
- [ ] LLM/Streamlit demo scoped as usability bonus, not a research claim.
