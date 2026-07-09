# PUBLICATION_ROADMAP.md — Venues, Strategy, Paper Outline

**Honest framing:** In one week on a laptop you can produce a **strong workshop paper** or a **mid-tier conference / journal short paper**. A NeurIPS/ICLR/ICML *main-track* accept is **not** realistic on this timeline or novelty budget — treat it as a later "v2" target after the workshop version and a full-data run. Plan for a two-stage publication: **v1 workshop/short paper now → v2 archival journal/conference later.**

---

## 1. Venue ranking (by realistic acceptance probability for the reframed project)

| Rank | Venue | Type | Scope fit | Difficulty | Novelty bar | Paper length | Timeline notes | Est. accept prob. (post-improvements) |
|---|---|---|---|---|---|---|---|---|
| 1 | **ML for Computational Biology (MLCB)** | Workshop/proceedings | Excellent (GNN + biology) | Low–Med | Moderate | 4–8 pp | Rolling / annual (often mid-year deadline) | **60–75%** |
| 2 | **NeurIPS / ICLR workshops** (AI4Science, MLSB, GenBio) | Workshop (non-archival) | Excellent | Low–Med | Moderate/idea-stage OK | 4–6 pp | Deadlines ~2–4 wks before the conf | **55–70%** |
| 3 | **IEEE BIBM** (Bioinformatics & Biomedicine) | Conference | Very good | Medium | Moderate | 6–8 pp IEEE | Annual, deadline ~Aug, conf Dec | **45–60%** |
| 4 | **ACM-BCB** (Bioinformatics, Comp. Bio, Health) | Conference | Very good | Medium | Moderate | 6–10 pp ACM | Annual, summer deadline | **45–55%** |
| 5 | **Bioinformatics / Bioinformatics Advances (OUP)** — Application Note / short | Journal | Excellent (tools+method) | Medium | Method+utility | ~2–4 pp (App Note) or full | Rolling | **40–55%** (needs clean reproducibility + utility) |
| 6 | **Journal of Proteome Research (ACS)** | Journal | Excellent (domain home) | Med–High | Solid method + validation | Full | Rolling | **35–50%** (rigorous domain review) |
| 7 | **ISMB/ECCB, RECOMB** | Conference | Excellent | High | High | Full | Winter deadlines | **15–30%** (stretch; v2) |
| 8 | **NeurIPS/ICLR/ICML/KDD main** | Conference | Good (needs ML novelty) | Very High | High | 8–9 pp | Fixed | **<15%** now (v2 goal) |

**Recommended primary target:** the best-timed **workshop (MLCB or a NeurIPS/ICLR AI4Science/MLSB workshop)** for v1, with **IEEE BIBM** as the archival submission if a deadline lands in range. **Verify all deadlines** before committing — they move yearly.

---

## 2. Submission strategy

1. **Anchor to Pep2Prob.** Position as "the missing GNN axis of the Pep2Prob benchmark," cite it prominently, use its exact splits/metrics. Reviewers who know the dataset will reward direct, fair comparison.
2. **Lead with the reframing + interpretability + efficiency**, not "we applied a GNN." The one-sentence pitch: *"We show peptide fragment-ion probability is best modeled as an edge-level cleavage-site task, giving an interpretable sub-1M-parameter GNN competitive with a Transformer on Pep2Prob and trainable on a laptop."*
3. **Pre-empt Reviewer #2.** Include: the global-pool-vs-edge ablation, the matched sequence baseline, significance tests, and the "graph earns its keep" edge ablation — exactly the objections in `PROJECT_REVIEW.md`.
4. **Reproducibility as a feature.** Ship code + configs + seeds + a one-command repro; mention the Streamlit demo as a usability bonus (not the contribution).
5. **Scope the LLM narration OUT** of the paper (it's a demo garnish, not science).
6. **Two-stage plan:** submit v1 to a workshop for feedback + timestamp; expand to full-608k training, multi-charge (235 channels), and a stronger interpretability study for a v2 archival submission (BIBM/JPR/ISMB).

---

## 3. Paper outline (6–8 pages)

1. **Abstract** — task, the edge-level reframing, headline result (SA vs Transformer, params, laptop), interpretability.
2. **Introduction** — fragment prediction matters for PSM scoring; probability (Pep2Prob) vs intensity; gap = no GNN / no cleavage-local model; contributions (from `RESEARCH_PLAN.md` §5).
3. **Related Work** — sequence models (Prosit, pDeep, AlphaPeptDeep, Alpha-Frag); GNNs for spectra (FIORA, MassFormer, ICEBERG); Pep2Prob benchmark; graph primitives.
4. **Background & biochemistry** — b/y ions, mobile-proton model, proline/acidic effects (motivates the design).
5. **Method** — graph construction, CleavageGNN, edge readout, charge features, loss.
6. **Experiments** — setup (splits/metrics), main table (vs Global/BoF/BiLSTM/Transformer), ablations, robustness, efficiency table.
7. **Analysis** — biochemical interpretability figures; error stratification; when structure helps.
8. **Limitations & Future Work** — subset training, charge-1 focus (if applicable), PTMs, ETD/CID transfer (mirror Pep2Prob's own limitations).
9. **Reproducibility statement + code/data availability.**

**Figures/tables to prepare:** (F1) architecture + edge-readout schematic; (F2) SA/L1 main comparison bar chart with error bars; (F3) proline/Asp motif recovery; (F4) attention/attribution heatmap on an example peptide; (T1) main results ±std; (T2) ablations; (T3) efficiency (params/VRAM/throughput).

---

## 4. Writing plan (maps to Day 7 in `MASTER_PLAN.md`)
- Draft Methods + Experiments **as you run** (Days 2–6), not at the end.
- Freeze results Day 6; write Intro/Related/Analysis/Abstract Day 7.
- Use the `PROJECT_REVIEW.md` weakness list as a "self-review" checklist before submission.

## 5. Realistic acceptance estimate after improvements
- **Workshop (MLCB / AI4Science / MLSB):** ~60–75%.
- **IEEE BIBM / ACM-BCB:** ~45–60%.
- **Bioinformatics App Note / JPR:** ~35–55% (contingent on reproducibility + a genuine utility or accuracy win).
- **Top ML main track:** <15% until a v2 with full-data results and clearer ML novelty.
