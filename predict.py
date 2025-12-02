import sys
import os

import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Make "src" importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.graph_builder import build_peptide_graph
from src.model import PepFragGNN
from src.llm_utils import interpret   # 🚀 NEW


# -------------------------------
# Constants
# -------------------------------
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


# -------------------------------
# Load Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PepFragGNN(
        in_dim=21,
        hidden_dim=128,
        num_layers=4,
        out_dim=78,
    ).to(device)

    state = torch.load("fragment_gnn.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, device


# -------------------------------
# Run prediction for a peptide
# -------------------------------
def predict(peptide, model, device):
    x, edge_index = build_peptide_graph(peptide)

    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = torch.zeros(x.shape[0], dtype=torch.long).to(device)

    with torch.no_grad():
        pred = model(x, edge_index, batch)  # [1, 78]

    return pred.squeeze(0).cpu().tolist()   # [78]


# -------------------------------
# Decode 78 outputs → b1 / y1
# -------------------------------
def decode_outputs(pred, pep_len):
    max_len = pep_len - 1
    b1 = pred[0:39][:max_len]
    y1 = pred[39:78][:max_len]
    return {"b1": b1, "y1": y1}


# -------------------------------
# Matplotlib / Plotly plots
# -------------------------------
def plot_b_y_series(df):
    fig, ax = plt.subplots()
    ax.plot(df["position"], df["b1"], marker="o", label="b1")
    ax.plot(df["position"], df["y1"], marker="o", label="y1")
    ax.set_xlabel("Fragment position")
    ax.set_ylabel("Probability")
    ax.set_title("Fragment Ion Probabilities (b1 / y1)")
    ax.set_xticks(df["position"])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mirror(df):
    fig, ax = plt.subplots()
    positions = df["position"]
    b_vals = df["b1"]
    y_vals = [-v for v in df["y1"]]
    ax.bar(positions, b_vals, width=0.4, align="center", label="b1 (top)")
    ax.bar(positions, y_vals, width=0.4, align="edge", label="y1 (bottom)")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Fragment position")
    ax.set_ylabel("Probability")
    ax.set_title("Mirror Plot: b1 (up) vs y1 (down)")
    ax.set_xticks(positions)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mirror_plotly(df, peptide):
    positions = df["position"].tolist()
    b_vals = df["b1"].tolist()
    y_vals = [-v for v in df["y1"].tolist()]
    fig = go.Figure()
    fig.add_bar(x=positions, y=b_vals, name="b1")
    fig.add_bar(x=positions, y=y_vals, name="y1 (mirrored)")
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(
        title=f"Interactive Mirror Plot — {peptide}",
        xaxis_title="Fragment position",
        yaxis_title="Probability",
        bargap=0.15,
        barmode="overlay",
    )
    return fig


def plot_heatmap(df, peptide):
    z = [df["b1"].tolist(), df["y1"].tolist()]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=df["position"].tolist(),
            y=["b1", "y1"],
            colorbar=dict(title="Probability"),
        )
    )
    fig.update_layout(
        title=f"Fragment Ion Heatmap — {peptide}",
        xaxis_title="Fragment position",
        yaxis_title="Ion type",
    )
    return fig


# -------------------------------
# Count model parameters
# -------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="Peptide Fragmentation GNN", page_icon="🧬", layout="wide")

    st.title("🧬 Peptide Fragmentation Predictor (GNN)")
    st.write("Graph Neural Network trained on **Pep2Prob** to predict b1/y1 fragmentation probabilities.")

    # Sidebar
    st.sidebar.header("Model")
    with st.sidebar:
        with st.spinner("Loading model..."):
            model, device = load_model()
        st.success(f"Model loaded on: `{device}`")

        total_params, trainable_params = count_parameters(model)
        st.caption(f"**Parameters:** {total_params:,} total ({trainable_params:,} trainable)")

        st.markdown("---")
        st.header("Input Help")
        st.markdown("Allowed AAs: A C D E F G H I K L M N P Q R S T V W Y")

    tab_single, tab_batch, tab_model = st.tabs(["🔹 Single Peptide", "📚 Batch Mode", "🧠 Model Info"])

    # ---------------------------------------------------------
    # SINGLE PEPTIDE TAB
    # ---------------------------------------------------------
    with tab_single:
        st.subheader("Single Peptide Prediction")

        peptide = st.text_input("Enter peptide sequence", value="PEPTIDER").strip().upper()
        run_single = st.button("🚀 Predict", type="primary")

        if run_single:
            if not peptide:
                st.error("Please enter a peptide sequence.")
                return

            invalid = [aa for aa in peptide if aa not in VALID_AAS]
            if invalid:
                st.error(f"Invalid amino acids: {set(invalid)}")
                return

            with st.spinner("Running prediction..."):
                pred = predict(peptide, model, device)
                decoded = decode_outputs(pred, len(peptide))

            with st.expander("Show raw 78 outputs (debug)"):
                st.write(pred)

            b1 = decoded["b1"]
            y1 = decoded["y1"]
            positions = list(range(1, len(peptide)))

            df = pd.DataFrame({"position": positions, "b1": b1, "y1": y1})

            # Summary
            st.markdown("### Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max b1", f"{max(b1):.3f}")
            c2.metric("Max y1", f"{max(y1):.3f}")
            c3.metric("Mean b1", f"{sum(b1)/len(b1):.3f}")
            c4.metric("Mean y1", f"{sum(y1)/len(y1):.3f}")

            # Skyline view
            st.markdown("### Skyline-style View")
            col_left, col_right = st.columns([1.1, 1.9])
            col_left.dataframe(df.style.format({"b1": "{:.4f}", "y1": "{:.4f}"}))
            col_right.plotly_chart(plot_mirror_plotly(df, peptide), use_container_width=True)

            # Additional visualizations
            st.markdown("### Additional Visualizations")
            s1, s2, s3 = st.tabs(["📈 Line Plot", "📉 Static Mirror", "🔥 Heatmap"])
            s1.pyplot(plot_b_y_series(df), clear_figure=True)
            s2.pyplot(plot_mirror(df), clear_figure=True)
            s3.plotly_chart(plot_heatmap(df, peptide))

            # Sequence view
            st.markdown("### Sequence View with Cuts")
            st.code(" ".join(list(peptide)))
            seq_df = pd.DataFrame({
                "cut_between": [f"{peptide[i]} | {peptide[i+1]}" for i in range(len(peptide)-1)],
                "position": positions,
                "b1": b1,
                "y1": y1,
            })
            st.dataframe(seq_df.style.format({"b1": "{:.4f}", "y1": "{:.4f}"}))

            # -------------------------------
            # ⭐ LLM Interpretation
            # -------------------------------
            st.markdown("### 🧠 AI Interpretation Settings")
            mode = st.radio(
                "Choose interpretation style:",
                ["Biologist-friendly (simple)", "Mass-spec Expert (technical)"],
                horizontal=True
            )
            mode_key = "biologist" if mode.startswith("Biologist") else "expert"

            with st.spinner("Generating AI Explanation..."):
                explanation = interpret(peptide, b1, y1, mode_key)

            st.markdown("### 🧬 LLM Interpretation")
            st.write(explanation)

            # Download CSV
            st.download_button(
                "💾 Download fragment table as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{peptide}_fragments.csv",
                mime="text/csv",
            )

    # ---------------------------------------------------------
    # MODEL INFO TAB
    # ---------------------------------------------------------
    with tab_model:
        st.subheader("Model Architecture & Details")
        st.json({
            "device": str(next(model.parameters()).device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "output_dim": 78,
            "hidden_dim": 128,
            "num_layers": 4,
        })


if __name__ == "__main__":
    main()
