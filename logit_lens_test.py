import torch
from get_model import get_model
from concept_vectors import load_concept_vectors

# Load model
print("Loading model...")
model, tokenizer = get_model()

# Determine layer (same as extraction - about 2/3 through)
num_layers = len(model.model.layers)
layer_idx = int(num_layers * 0.66)
print(f"Using layer {layer_idx} out of {num_layers}")

# Load concept vectors
print("\nLoading concept vectors...")
vectors, labels = load_concept_vectors("concept_vectors.pt")

sadness_idx = labels.index("sugar")
sadness_vec = vectors[sadness_idx]
normalized = model.model.norm(sadness_vec.unsqueeze(0).to(model.device)) 
logits = model.lm_head(normalized.to(model.device))  # Project to vocabulary

top_tokens = logits.topk(20)
print("Top tokens:")
for token_id in top_tokens.indices[0]:
    print(tokenizer.decode([token_id]))

