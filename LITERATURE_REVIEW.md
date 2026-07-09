# LITERATURE_REVIEW.md — Grounding, Related Work, and Research Gaps

**Scope:** Peptide MS/MS fragment prediction (intensity & probability), GNNs for spectrum/fragmentation prediction, the biochemistry of peptide dissociation, and the graph-learning primitives you will use.
**How to read this:** Each entry gives citation → link → main idea → relevance → gap it leaves → how you differ → cite? (Y/N). Categories are ordered by importance to your paper.

> Reviewer note: citation years/authors below are given to the best of current knowledge — **verify each on the linked page and in the actual PDF before adding to the bibliography.** Do not cite anything you have not opened.

---

## A. The dataset you use — cite ALL, this is non-negotiable

### A1. Pep2Prob Benchmark (2025) — **your dataset and your direct competitor**
- arXiv: https://arxiv.org/abs/2508.21076 · HTML: https://arxiv.org/html/2508.21076v1 · OpenReview: https://openreview.net/forum?id=A5MPzwyq0H · Data: https://huggingface.co/datasets/bandeiralab/Pep2Prob
- **Main idea:** First large-scale dataset + benchmark for *peptide-specific fragment-ion probability*. 608,780 precursors from 183M HCD MS2 spectra; lengths 7–40; precursor charge 1–8; 235 fragment ions (1 a, 117 b, 117 y at charges 1–3). Probability = fraction of spectra where fragment intensity > ε=1e-6. Leakage-controlled split by graph connected components (identical sequences or shared 6-mer prefix/suffix), 5 folds. Baselines: Global, Bag-of-Fragment, Linear Regression, ResNet, Transformer. **Best (Transformer): L1 0.056, MSE 0.017, Spectral Angle 0.845, Acc 0.953.** Metrics: L1, MSE, Spectral Angle (type-1); Accuracy/Sensitivity/Specificity at τ=0.001 (type-2).
- **Relevance:** This *is* your problem, dataset, split, metrics, and SOTA. Your paper is defined relative to this one.
- **Gap it leaves:** (1) No **graph neural network** among baselines — their "graph" is only for the data split, not the model. (2) Their Transformer is **sequence/token-centric**; no **cleavage-site (bond/edge) locality** inductive bias. (3) No **biochemical interpretability** analysis (proline effect, charge retention). (4) No **parameter-efficiency / laptop-scale** study.
- **How you differ:** You add the missing GNN axis with an *edge-centric* readout aligned to the physical cleavage event, plus interpretability and efficiency.
- **Cite? YES — foreground, compare directly, use their exact metrics and splits.**

---

## B. Foundational peptide MS/MS fragment prediction (the lineage you extend)

### B1. Prosit — Gessulat et al., *Nature Methods* 2019
- https://www.nature.com/articles/s41592-019-0426-7 · PMID 31133760
- **Main idea:** Encoder–decoder (GRU/LSTM, attention) predicting fragment-ion **intensities** + retention time from sequence + charge + collision energy; trained on ProteomeTools (~550k peptides, 21M spectra). Improved DB-search FDR by >10×.
- **Relevance:** The canonical deep-learning fragment predictor; defines the sequence-model paradigm you contrast against.
- **Gap:** Predicts *intensity* not *probability*; sequence RNN, not graph; large model.
- **You differ:** Probability task (Pep2Prob), graph/edge model, laptop-scale.
- **Cite? YES (foundational).**

### B2. pDeep / pDeep2 / pDeep3 — Zhou et al. 2017 (*Anal. Chem.*), Zeng et al. 2019, Ching et al. 2021
- pDeep2: https://pubs.acs.org/doi/10.1021/acs.analchem.9b01262 (verify) · search "pDeep MS/MS BiLSTM"
- **Main idea:** BiLSTM predicting b/y (and neutral-loss) intensities; pDeep2 adds transfer learning for modified peptides.
- **Relevance:** Second pillar of the sequence-model lineage; introduces modification handling and transfer learning you may borrow.
- **Gap:** Intensity, RNN, no graph, no probability task.
- **Cite? YES.**

### B3. AlphaPeptDeep — Zeng et al., *Nature Communications* 2022
- https://www.nature.com/articles/s41467-022-34904-3 · PMC9700817
- **Main idea:** Modular transformer framework for peptide properties (MS2 intensity, RT, CCS). **Key stat for you: 4 M-parameter MS2 model vs 64 M Prosit-Transformer, ~40× faster** with comparable accuracy.
- **Relevance:** **Your efficiency north-star.** It legitimizes "small model, laptop, competitive accuracy" as a publishable framing.
- **Gap:** Transformer, intensity, no probability task, no graph, no cleavage-locality bias.
- **You differ:** Even smaller GNN, probability task, interpretable edge readout.
- **Cite? YES — use as the efficiency benchmark rhetoric.**

### B4. Alpha-Frag — Ai et al., 2021 (bioRxiv/*MCP*) — **closest task to yours**
- https://www.biorxiv.org/content/10.1101/2021.04.07.438629
- **Main idea:** DNN predicting **fragment presence/probability** (not intensity) to improve identification.
- **Relevance:** The nearest prior *task* (existence/probability rather than intensity). Direct conceptual neighbor to Pep2Prob and to you.
- **Gap:** Pre-Pep2Prob (smaller/older data), sequence DNN, no graph, no edge locality.
- **Cite? YES — position as prior fragment-*presence* work.**

### B5. MS2PIP — Degroeve, Martens et al., *Nucleic Acids Research* 2015/2019/2023
- 2019 server: https://academic.oup.com/nar/article/47/W1/W295/5480903 (verify)
- **Main idea:** Gradient-boosted trees (XGBoost) predicting peak intensities from engineered features; fast, CPU-friendly, widely used.
- **Relevance:** The classic **feature-based, non-deep** baseline; strong "simple model" comparator and a source of hand-crafted features (flanking residues, position, basicity) you can feed a GNN.
- **Gap:** Not deep, not graph, intensity not probability.
- **Cite? YES — as classical baseline + feature inspiration.**

### B6. DeepMass:Prism — Tiwary et al., *Nature Methods* 2019
- https://www.nature.com/articles/s41592-019-0427-6 (verify)
- **Main idea:** Deep model for spectral prediction to build in-silico libraries.
- **Relevance:** Contemporary of Prosit; completes the 2019 deep-prediction picture.
- **Cite? Optional (Y if space) — related work breadth.**

### B7. Systematic assessment of DL fragmentation predictors — Declercq et al., *J. Proteome Res.* 2024
- https://pubs.acs.org/doi/10.1021/acs.jproteome.3c00857 · PMC11165591
- **Main idea:** Head-to-head evaluation of Prosit/MS2PIP/pDeep/AlphaPeptDeep, standardizing metrics (incl. spectral angle) and pitfalls.
- **Relevance:** Your **methodological rulebook** for fair fragment-prediction evaluation; cite to justify metric choices and avoid known evaluation traps.
- **Cite? YES.**

---

## C. GNNs for mass spectra / fragmentation (your architectural family — mostly small molecules, = the gap)

### C1. FIORA — Nowatzky et al., *Nature Communications* 2025 — **your strongest methodological analogue**
- https://www.nature.com/articles/s41467-025-57422-4 · PMC11889238
- **Main idea:** GNN predicting tandem spectra by learning the **molecular neighborhood of each bond** to derive fragment-ion probabilities; beats ICEBERG and CFM-ID.
- **Relevance:** **Proves the exact inductive bias you propose** — *bond/edge-local* message passing → fragment probability — but for **small molecules/metabolites**, not peptides.
- **Gap:** Not peptides; no peptide-specific charge/mobile-proton modeling; not Pep2Prob.
- **You differ:** Transfer the bond-neighborhood → probability principle to **peptide backbones** on **Pep2Prob**, with peptide-specific edges/features.
- **Cite? YES — this is your "we adapt the idea that worked for small molecules to peptides" anchor.**

### C2. MassFormer — Young et al., 2021/2024 (*JMLR*/Nat. Mach. Intell., verify)
- https://arxiv.org/abs/2111.04824
- **Main idea:** Graph Transformer for small-molecule MS/MS spectrum prediction; first transformer on this task.
- **Relevance:** Graph-transformer precedent for spectrum prediction; motivates a graph-attention variant of your model.
- **Gap:** Small molecules; graph-level, not cleavage-local; not peptides.
- **Cite? YES.**

### C3. ICEBERG & SCARF — Goldman et al., 2023/2024 (`ms-pred`)
- https://github.com/coleygroup/ms-pred
- **Main idea:** Physically-grounded, autoregressive fragmentation-graph generation + intensity scoring for small molecules.
- **Relevance:** State-of-the-art structured fragmentation modeling; conceptual contrast (generative fragmentation graph vs your direct edge regression).
- **Gap:** Small molecules; heavier machinery; not peptides/Pep2Prob.
- **Cite? YES (related work).**

### C4. CFM-ID — Allen et al., 2014+ / Wang et al. 2021
- https://cfmid.wishartlab.com/ (verify paper)
- **Main idea:** Probabilistic (Markov) small-molecule fragmentation model; pre-deep-learning standard.
- **Relevance:** Classical fragmentation-probability modeling; historical grounding.
- **Cite? Optional (Y for completeness).**

---

## D. Graph-learning primitives (cite the ones you actually use)

### D1. GCN — Kipf & Welling, *ICLR* 2017 — https://arxiv.org/abs/1609.02907 — your current backbone. **Cite? YES.**
### D2. GraphSAGE — Hamilton et al., *NeurIPS* 2017 — https://arxiv.org/abs/1706.02216 — inductive aggregation. **Cite? if used.**
### D3. GAT — Veličković et al., *ICLR* 2018 — https://arxiv.org/abs/1710.10903 — attention over neighbors; natural for weighting charge-carrier residues. **Cite? if used (recommended variant).**
### D4. GIN — Xu et al., *ICLR* 2019 ("How Powerful are GNNs?") — https://arxiv.org/abs/1810.00826 — expressive power / WL bound; justifies replacing GCN with a more expressive layer. **Cite? YES (motivates architecture choice).**
### D5. Graphormer — Ying et al., *NeurIPS* 2021 — https://arxiv.org/abs/2106.05667 — graph + positional/structural encodings; bridges to MassFormer. **Cite? if you add structural encodings.**
### D6. Message Passing Neural Networks — Gilmer et al., *ICML* 2017 — https://arxiv.org/abs/1704.01212 — the MPNN framework and edge-conditioned readouts you generalize. **Cite? YES.**

---

## E. Biochemical grounding (turns your model into science, not curve-fitting)

### E1. Mobile proton model — Wysocki, Tsaprailis, Smith, Breci, *J. Mass Spectrom.* 2000
- https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/1096-9888(200012)35:12%3C1399::AID-JMS86%3E3.0.CO;2-R · PMID 11180630
- **Main idea:** Charge-directed dissociation: proton mobility (governed by basic residues R/K/H and precursor charge) controls where the backbone cleaves and thus b/y intensities.
- **Relevance:** **The physical theory your interpretability section tests.** Predicts: charge retention at basic residues, suppression under low proton mobility. Motivates charge features and charge-carrier edges.
- **Cite? YES — frame your feature design and error analysis around it.**

### E2. Proline effect / selective cleavage — Breci et al. 2003; Huang et al.; comprehensive Pro CID study (*IJMS* 2011)
- e.g. https://www.sciencedirect.com/science/article/abs/pii/S1387380611003095
- **Main idea:** Enhanced cleavage **N-terminal to Pro**, suppressed **C-terminal to Pro**; enhanced cleavage **C-terminal to Asp/Glu** under low mobility.
- **Relevance:** Concrete, testable motifs for your qualitative evaluation — "does the GNN recover the proline effect?" is a compelling figure.
- **Cite? YES.**

### E3. Spectral angle / normalized spectral contrast angle — Toprak et al. 2014 / used in Prosit & Pep2Prob
- **Main idea:** The standard similarity metric between predicted and observed spectra (1 − arccos-based angle).
- **Relevance:** Your primary metric; cite the origin so reviewers accept it.
- **Cite? YES.**

---

## F. Research gaps → your opportunities (the punchline)

1. **No GNN baseline on Pep2Prob.** The benchmark's own graph usage is only for splitting; the *model* axis of "graphs" is empty. You fill it. *(Primary novelty.)*
2. **No cleavage-site (edge-level) formulation for peptides.** FIORA proves bond-local prediction works for small molecules; nobody has done the peptide-backbone analogue on probability data. *(Primary novelty.)*
3. **No biochemical interpretability study on Pep2Prob.** Whether a learned model recovers mobile-proton / proline / Asp effects is open and cheap to test. *(Secondary novelty — high novelty-to-effort.)*
4. **No efficiency/parameter study.** AlphaPeptDeep legitimized "small & fast"; a <1 M-parameter GNN matching a Transformer at a fraction of the cost, trainable on a 6 GB laptop, is a defensible systems angle. *(Secondary novelty.)*
5. **Charge-conditioning and multi-charge fragments under-modeled** relative to the dataset's full 235-channel, charge-1–8 scope. Extending beyond charge-1 b/y is low-risk incremental novelty.
6. **Loss-function mismatch.** No one has studied which objective (BCE vs MSE vs spectral-angle loss) best optimizes the SA metric on this data — a small, clean ablation contribution.

**Recommended novelty stack (feasible in a week, ranked by novelty-to-effort):**
`Edge-centric GNN (C1-inspired)` → `charge/mobile-proton features (E1)` → `biochemical interpretability (E1/E2)` → `efficiency vs Transformer (B3)` → `loss ablation (F6)`.

---

## G. Minimum citation set for the paper (~18)
A1 (Pep2Prob), B1 (Prosit), B2 (pDeep), B3 (AlphaPeptDeep), B4 (Alpha-Frag), B5 (MS2PIP), B7 (systematic assessment), C1 (FIORA), C2 (MassFormer), C3 (ICEBERG/SCARF), D1 (GCN), D3 (GAT), D4 (GIN), D6 (MPNN), E1 (mobile proton), E2 (proline effect), E3 (spectral angle). Add D5/C4/B6 if space.

**Sources consulted:** [Pep2Prob](https://arxiv.org/abs/2508.21076) · [Prosit](https://www.nature.com/articles/s41592-019-0426-7) · [AlphaPeptDeep](https://www.nature.com/articles/s41467-022-34904-3) · [FIORA](https://www.nature.com/articles/s41467-025-57422-4) · [MassFormer](https://arxiv.org/abs/2111.04824) · [ms-pred/ICEBERG](https://github.com/coleygroup/ms-pred) · [Alpha-Frag](https://www.biorxiv.org/content/10.1101/2021.04.07.438629) · [Systematic assessment](https://pubs.acs.org/doi/10.1021/acs.jproteome.3c00857) · [Mobile proton model](https://pubmed.ncbi.nlm.nih.gov/11180630/) · [Proline effect](https://www.sciencedirect.com/science/article/abs/pii/S1387380611003095) · [GCN](https://arxiv.org/abs/1609.02907) · [GIN](https://arxiv.org/abs/1810.00826)
