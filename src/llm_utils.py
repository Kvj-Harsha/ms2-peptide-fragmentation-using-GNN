import os
from groq import Groq

# ----------------------------------------------------------
# 🔐 DIRECT API KEY (paste your key here)
# ----------------------------------------------------------
GROQ_API_KEY = "put-your-own-key"

# ----------------------------------------------------------
# ✨ Updated Model Choices
# ----------------------------------------------------------
MODEL_BIO = "llama-3.3-70b-versatile"     # simple explanations
MODEL_EXPERT = "qwen/qwen3-32b"           # deep technical explanations


# ----------------------------------------------------------
# Groq Client
# ----------------------------------------------------------
def get_client():
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ No Groq API key found! Set GROQ_API_KEY or paste it in llm_utils.py.")
    return Groq(api_key=api_key)


# ----------------------------------------------------------
# Build Prompts
# ----------------------------------------------------------
def build_biologist_prompt(peptide, b1, y1):
    return f"""
Explain this MS/MS fragmentation pattern like a friendly molecular biologist.

Peptide: {peptide}
Length: {len(peptide)} aa
Fragment positions: {len(b1)}

b-ion (N-terminal) probabilities:
{b1}

y-ion (C-terminal) probabilities:
{y1}

Explain simply:
- Which region breaks strongly (start/middle/end)
- Whether fragments are strong or weak
- How b vs y ions compare
- What this means for identifying this peptide

Avoid jargon. Keep it intuitive.
"""


def build_expert_prompt(peptide, b1, y1):
    return f"""
Provide a full expert-level proteomics interpretation (mass spectrometry).

Peptide: {peptide}
Length: {len(peptide)} aa
Fragment positions: {len(b1)}

b1 ion probabilities:
{b1}

y1 ion probabilities:
{y1}

Explain technically:
- Fragmentation hot/cold spots
- Proton mobility and charge retention effects
- Sequence-driven cleavage (Ala/Gly suppression, aromatic boosts, Pro block)
- Expected MS2 spectral quality
- Relative strength of b vs y ladder
- Expected search engine scoring behavior (e.g., SEQUEST/Andromeda)
- Whether the peptide is easy or difficult to identify

Be highly scientific and detailed.
"""


# ----------------------------------------------------------
# Main Interpretation Function
# ----------------------------------------------------------
def interpret(peptide, b1, y1, mode="biologist"):
    client = get_client()

    prompt = (
        build_biologist_prompt(peptide, b1, y1)
        if mode == "biologist"
        else build_expert_prompt(peptide, b1, y1)
    )

    response = client.chat.completions.create(
        model=MODEL_BIO if mode == "biologist" else MODEL_EXPERT,
        messages=[
            {"role": "system", "content": "You analyze peptide fragmentation patterns."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        temperature=0.25,
    )

    # FIXED LINE:
    return response.choices[0].message.content
