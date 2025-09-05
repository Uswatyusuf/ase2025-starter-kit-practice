from transformers import AutoTokenizer, T5EncoderModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import normalize

# 2) Load the encoder‐only model & tokenizer
model_id   = "Salesforce/codet5p-770m"
tokenizer  = AutoTokenizer.from_pretrained(model_id)
model      = T5EncoderModel.from_pretrained(model_id)  # <- encoder only

# 3) Example prefix & suffix around your missing code
prefix = "def compute_factorial(n):\n    if n == 0:\n        return 1\n    else:\n        "
suffix = "\n\n# end of function"
query  = prefix + tokenizer.sep_token + suffix

# 4) One candidate snippet
candidate = "    return n * compute_factorial(n - 1)"

# 5) Embedding helper
def embed_mean(text):
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       max_length=4096)
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state  # [1, T, D]
    mask    = inputs.attention_mask.unsqueeze(-1)  # [1, T, 1]
    summed  = (hidden * mask).sum(1)               # [1, D]
    counts  = mask.sum(1).clamp(min=1)             # [1, 1]
    emb     = (summed / counts).cpu().numpy()      # [1, D]
    return normalize(emb)                          # L2‑normalized

# Query
q_emb = (embed_mean(prefix) + embed_mean(suffix)) / 2
# Candidate
c_emb = embed_mean(candidate)

# Cosine similarity
sim = float((q_emb * c_emb).sum())
print(f"Similarity score: {sim:.4f}")
