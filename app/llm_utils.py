"""Optional LLM narration for the demo. Key comes ONLY from the environment.

Set GROQ_API_KEY in your shell (or a .env you do not commit) to enable this.
No key is ever stored in source. The narration is a usability bonus for the
demo, not a research claim.
"""
from __future__ import annotations

import os

MODEL_SIMPLE = "llama-3.3-70b-versatile"
MODEL_EXPERT = "qwen/qwen3-32b"


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it in your environment to enable "
            "LLM narration; it is intentionally never read from source."
        )
    from groq import Groq  # imported lazily so the app runs without groq installed
    return Groq(api_key=api_key)


def _prompt(peptide, sites, mode):
    rows = "\n".join(
        f"  site {k+1} ({peptide[k]}|{peptide[k+1]}): b={b:.3f} y={y:.3f}"
        for k, (b, y) in enumerate(sites)
    )
    if mode == "expert":
        ask = ("Give an expert proteomics interpretation: fragmentation hot/cold "
               "spots, proton mobility and charge retention, proline/acidic-residue "
               "effects, expected MS2 quality and identifiability.")
    else:
        ask = ("Explain simply which regions break, whether fragments are strong or "
               "weak, how b vs y compare, and what it means for identifying this peptide.")
    return (f"Peptide: {peptide} (length {len(peptide)})\n"
            f"Per-cleavage-site b/y probabilities:\n{rows}\n\n{ask}")


def interpret(peptide, sites, mode="simple"):
    """sites: list of (b_prob, y_prob) per cleavage site. Returns model text."""
    client = _get_client()
    model = MODEL_EXPERT if mode == "expert" else MODEL_SIMPLE
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You analyze peptide fragmentation patterns."},
            {"role": "user", "content": _prompt(peptide, sites, mode)},
        ],
        max_tokens=600,
        temperature=0.25,
    )
    return resp.choices[0].message.content
