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

# -------------------------------
# Constants
# -------------------------------
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


# -------------------------------
# Load Model (cached)
# NOTE: hidden_dim=128, num_layers=4
#       so it matches your trained weights
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
    """First 39 = b1; next 39 = y1"""
    max_len = pep_len - 1  # valid fragment positions

    b1 = pred[0:39][:max_len]
    y1 = pred[39:78][:max_len]

    return {"b1": b1, "y1": y1}


# -------------------------------
# Matplotlib line plot (b1/y1)
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


# -------------------------------
# Matplotlib mirror plot
# -------------------------------
def plot_mirror(df):
    fig, ax = plt.subplots()
    positions = df["position"]

    b_vals = df["b1"]
    y_vals = [-v for v in df["y1"]]   # flip y-ions

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


# -------------------------------
# Plotly mirror plot (interactive)
# -------------------------------
def plot_mirror_plotly(df, peptide):
    positions = df["position"].tolist()
    b_vals = df["b1"].tolist()
    y_vals = [-v for v in df["y1"].tolist()]  # negative for mirror

    fig = go.Figure()

    fig.add_bar(
        x=positions,
        y=b_vals,
        name="b1",
        hovertemplate="b<sub>%{x}</sub>: %{y:.4f}<extra></extra>",
    )

    fig.add_bar(
        x=positions,
        y=y_vals,
        name="y1 (mirrored)",
        hovertemplate="y<sub>%{x}</sub>: %{customdata:.4f}<extra></extra>",
        customdata=df["y1"].tolist(),
    )

    fig.add_hline(y=0, line_width=1)

    fig.update_layout(
        title=f"Interactive Mirror Plot — {peptide}",
        xaxis_title="Fragment position",
        yaxis_title="Probability (b1 up, y1 down)",
        bargap=0.15,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# -------------------------------
# Plotly heatmap (b1 + y1)
# -------------------------------
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
# Model parameter counting
# -------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config(
        page_title="Peptide Fragmentation GNN",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 Peptide Fragmentation Predictor (GNN)")
    st.write(
        "Graph Neural Network trained on **Pep2Prob** to predict "
        "fragment ion probabilities (b1 / y1) for a given peptide."
    )

    # Sidebar -------------------------------------------------
    st.sidebar.header("Model")
    with st.sidebar:
        with st.spinner("Loading model..."):
            model, device = load_model()
        st.success(f"Model loaded on: `{device}`")

        total_params, trainable_params = count_parameters(model)
        st.caption(
            f"**Parameters:** {total_params:,} total "
            f"({trainable_params:,} trainable)"
        )

        st.markdown("---")
        st.header("Input Help")
        st.markdown(
            """
            **Allowed AAs**  
            A, C, D, E, F, G, H, I, K, L,  
            M, N, P, Q, R, S, T, V, W, Y
            """
        )

    # Tabs ----------------------------------------------------
    tab_single, tab_batch, tab_model = st.tabs(
        ["🔹 Single Peptide", "📚 Batch Mode", "🧠 Model Info"]
    )

    # ---------------------------------------------------------
    # SINGLE PEPTIDE TAB
    # ---------------------------------------------------------
    with tab_single:
        st.subheader("Single Peptide Prediction")

        col_input, col_example = st.columns([2, 1])

        with col_input:
            peptide = st.text_input(
                "Enter peptide sequence",
                value="PEPTIDER",
                help="Use standard one-letter amino acid codes.",
            ).strip().upper()

        with col_example:
            st.write("Examples:")
            st.code("PEPTIDER\nACDEFGHIK\nMNPQRSTVWY")

        run_single = st.button("🚀 Predict", type="primary")

        if run_single:
            # Validation
            if not peptide:
                st.error("Please enter a peptide sequence.")
                return

            invalid = [aa for aa in peptide if aa not in VALID_AAS]
            if invalid:
                st.error(
                    f"Invalid amino acids found: {set(invalid)}. "
                    "Allowed: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y."
                )
                return

            if len(peptide) < 2:
                st.error("Peptide must be at least length 2 (need at least one fragment).")
                return

            with st.spinner("Running prediction..."):
                pred = predict(peptide, model, device)
                decoded = decode_outputs(pred, len(peptide))

            # Raw outputs
            with st.expander("Show raw 78 outputs (debug)"):
                st.write(pred)

            b1 = decoded["b1"]
            y1 = decoded["y1"]
            positions = list(range(1, len(peptide)))  # 1..L-1

            df = pd.DataFrame(
                {
                    "position": positions,
                    "b1": b1,
                    "y1": y1,
                }
            )

            # Quick stats
            st.markdown("### Summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Max b1", f"{max(b1):.3f}")
            with c2:
                st.metric("Max y1", f"{max(y1):.3f}")
            with c3:
                st.metric("Mean b1", f"{sum(b1)/len(b1):.3f}")
            with c4:
                st.metric("Mean y1", f"{sum(y1)/len(y1):.3f}")

            # SKYLINE-STYLE LAYOUT -----------------------
            st.markdown("### Skyline-style View")
            col_left, col_right = st.columns([1.1, 1.9])

            with col_left:
                st.caption("Fragment table")
                st.dataframe(
                    df.style.format({"b1": "{:.4f}", "y1": "{:.4f}"}),
                    use_container_width=True,
                    height=300,
                )

            with col_right:
                st.caption("Mirror plot (interactive)")
                fig_mirror_interactive = plot_mirror_plotly(df, peptide)
                st.plotly_chart(fig_mirror_interactive, use_container_width=True)

            # EXTRA VISUALS ------------------------------
            st.markdown("### Additional Visualizations")

            subtab_line, subtab_mirror, subtab_heat = st.tabs(
                ["📈 Line Plot", "📉 Static Mirror", "🔥 Heatmap"]
            )

            with subtab_line:
                fig_line = plot_b_y_series(df)
                st.pyplot(fig_line, clear_figure=True)

            with subtab_mirror:
                fig_mirror = plot_mirror(df)
                st.pyplot(fig_mirror, clear_figure=True)

            with subtab_heat:
                fig_heat = plot_heatmap(df, peptide)
                st.plotly_chart(fig_heat, use_container_width=True)

            # Sequence-aware view
            st.markdown("### Sequence View with Cuts")
            st.write(f"**Sequence ({len(peptide)} aa):**")
            st.code(" ".join(list(peptide)), language="text")

            seq_df = pd.DataFrame(
                {
                    "cut_between": [
                        f"{peptide[i]} | {peptide[i+1]}"
                        for i in range(len(peptide) - 1)
                    ],
                    "position": positions,
                    "b1": b1,
                    "y1": y1,
                }
            )
            st.dataframe(
                seq_df.style.format({"b1": "{:.4f}", "y1": "{:.4f}"}),
                use_container_width=True,
            )

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download fragment table as CSV",
                data=csv,
                file_name=f"{peptide}_fragments.csv",
                mime="text/csv",
            )

    # ---------------------------------------------------------
    # BATCH MODE TAB
    # ---------------------------------------------------------
    with tab_batch:
        st.subheader("Batch Prediction Mode")

        st.markdown(
            "Paste **one peptide per line** below and run predictions "
            "for all of them. Useful for small-scale benchmarking."
        )

        default_peps = "PEPTIDER\nACDEFGHIK\nMNPQRSTVWY"
        text_input = st.text_area(
            "Peptide list",
            value=default_peps,
            height=150,
        )

        run_batch = st.button("🚀 Run Batch Predictions")

        if run_batch:
            peptides = [
                p.strip().upper()
                for p in text_input.splitlines()
                if p.strip()
            ]
            peptides = list(dict.fromkeys(peptides))  # unique, preserve order

            if not peptides:
                st.error("Please provide at least one peptide.")
            else:
                valid_rows = []
                invalid_peps = []

                with st.spinner("Running batch predictions..."):
                    for pep in peptides:
                        if any(aa not in VALID_AAS for aa in pep) or len(pep) < 2:
                            invalid_peps.append(pep)
                            continue

                        pred = predict(pep, model, device)
                        decoded = decode_outputs(pred, len(pep))
                        b1 = decoded["b1"]
                        y1 = decoded["y1"]
                        positions = list(range(1, len(pep)))

                        for pos, b, y in zip(positions, b1, y1):
                            valid_rows.append(
                                {
                                    "peptide": pep,
                                    "position": pos,
                                    "b1": b,
                                    "y1": y,
                                }
                            )

                if invalid_peps:
                    st.warning(
                        "Some peptides were skipped (invalid or too short): "
                        + ", ".join(invalid_peps)
                    )

                if not valid_rows:
                    st.error("No valid peptides to display.")
                else:
                    result_df = pd.DataFrame(valid_rows)
                    st.markdown("### Per-fragment Results")
                    st.dataframe(
                        result_df.head(500).style.format(
                            {"b1": "{:.4f}", "y1": "{:.4f}"}
                        ),
                        use_container_width=True,
                        height=400,
                    )

                    # Per-peptide summary
                    summary = (
                        result_df.groupby("peptide")
                        .agg(
                            mean_b1=("b1", "mean"),
                            mean_y1=("y1", "mean"),
                            max_b1=("b1", "max"),
                            max_y1=("y1", "max"),
                            n_frags=("position", "count"),
                        )
                        .reset_index()
                    )
                    st.markdown("### Per-peptide Summary")
                    st.dataframe(
                        summary.style.format(
                            {
                                "mean_b1": "{:.4f}",
                                "mean_y1": "{:.4f}",
                                "max_b1": "{:.4f}",
                                "max_y1": "{:.4f}",
                            }
                        ),
                        use_container_width=True,
                    )

                    # Download full results
                    csv_all = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Download all fragment results (CSV)",
                        data=csv_all,
                        file_name="batch_fragments.csv",
                        mime="text/csv",
                    )

    # ---------------------------------------------------------
    # MODEL INFO TAB
    # ---------------------------------------------------------
    with tab_model:
        st.subheader("Model Architecture & Details")

        st.markdown(
            """
            **Architecture**

            - Backbone: Graph Convolutional Network (GCNConv, PyTorch Geometric)  
            - Input: peptide as a graph (residues as nodes, peptide bonds as edges)  
            - Node features: 21-dim AA one-hot (including special)  
            - Readout: graph-level mean pooling  
            - Output: 78-dim vector  
                - First 39 = b1 probabilities  
                - Next 39 = y1 probabilities  
            """
        )

        total_params, trainable_params = count_parameters(model)
        st.json(
            {
                "device": str(next(model.parameters()).device),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "output_dim": 78,
                "hidden_dim": 128,
                "num_layers": 4,
            }
        )

        with st.expander("Show model `state_dict` keys (debug)"):
            keys = list(model.state_dict().keys())
            st.write(keys)


if __name__ == "__main__":
    main()
