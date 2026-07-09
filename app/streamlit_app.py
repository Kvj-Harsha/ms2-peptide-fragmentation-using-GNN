"""Streamlit demo for CleavageGNN edge-level fragment-probability prediction.

Run:  streamlit run app/streamlit_app.py -- --ckpt runs/cleavage_gcn/model.pt
The demo (and optional LLM narration) is a usability bonus, not a research claim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import torch
from torch_geometric.loader import DataLoader

from pepfraggnn.config import Config
from pepfraggnn.data.dataset import make_synthetic_dataset
from pepfraggnn.engine import unpack_batch
from pepfraggnn.models import build_model
from pepfraggnn.seed import resolve_device
from pepfraggnn.utils import count_parameters

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _parse_ckpt() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/cleavage_gcn/model.pt")
    args, _ = parser.parse_known_args()
    return args.ckpt


@st.cache_resource
def load_model(ckpt_path: str):
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = Config.from_dict(payload["config"])
    device = resolve_device(cfg.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg, device


def predict(model, cfg, device, peptide):
    ds = make_synthetic_dataset([peptide], cfg=cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            logits, _, _, _ = unpack_batch(out, data, cfg.model.readout)
            return torch.sigmoid(logits).cpu()


def main():
    st.set_page_config(page_title="CleavageGNN", page_icon="🧬", layout="wide")
    st.title("🧬 CleavageGNN — Peptide Fragment-Ion Probability")
    st.caption("Edge-level GNN predicting per-cleavage-site b/y probabilities on Pep2Prob.")

    ckpt_path = _parse_ckpt()
    if not Path(ckpt_path).exists():
        st.error(f"Checkpoint not found: {ckpt_path}\n\nTrain one with "
                 "`python scripts/train.py` or pass `-- --ckpt <path>`.")
        return

    model, cfg, device = load_model(ckpt_path)
    total, trainable = count_parameters(model)
    st.sidebar.success(f"Model on `{device}`")
    st.sidebar.caption(f"{total:,} params ({trainable:,} trainable) | "
                       f"{cfg.model.readout}/{cfg.model.backbone}")

    peptide = st.text_input("Peptide sequence", value="PEPTIDER").strip().upper()
    if st.button("Predict", type="primary") and peptide:
        invalid = sorted(set(peptide) - VALID_AA)
        if invalid:
            st.warning(f"Non-canonical residues treated as UNK: {invalid}")

        probs = predict(model, cfg, device, peptide)
        L = len(peptide)
        rows = [{
            "site": k,
            "bond": f"{peptide[k-1]} | {peptide[k]}",
            "b": probs[k - 1, 0].item(),
            "y": probs[k - 1, 1].item(),
        } for k in range(1, L)]
        df = pd.DataFrame(rows).set_index("site")

        c1, c2 = st.columns([1, 2])
        c1.dataframe(df.style.format({"b": "{:.4f}", "y": "{:.4f}"}))
        c2.bar_chart(df[["b", "y"]])

        st.download_button("Download CSV", df.to_csv().encode("utf-8"),
                           file_name=f"{peptide}_fragments.csv", mime="text/csv")

        with st.expander("🧠 LLM narration (optional, needs GROQ_API_KEY)"):
            mode = st.radio("Style", ["simple", "expert"], horizontal=True)
            if st.button("Explain"):
                try:
                    from llm_utils import interpret
                    sites = [(r["b"], r["y"]) for r in rows]
                    st.write(interpret(peptide, sites, mode))
                except Exception as e:  # noqa: BLE001 - surface config errors to user
                    st.info(str(e))


if __name__ == "__main__":
    main()
