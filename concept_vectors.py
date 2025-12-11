import torch
from get_model import get_model


def extract_concept_vectors(
    model,
    tokenizer,
    concept_words,
    layer_idx,
    batch_size=8,
):
    prompt = "Tell me about {word}."


    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output[0].detach()
        return hook
    
    model.model.layers[layer_idx].register_forward_hook(get_activation('target_activation'))
    activations_list = []
    for i, word in enumerate(concept_words):
        assert len(activation) == 0, "Activations dict should be empty before forward pass."
        message = [{"role": "user", "content": prompt.format(word=word)}]
        text = tokenizer.apply_chat_template(
            message,
            tokenize = False,
            add_generation_prompt = True
        ) 
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**model_inputs)
        activations_list.append(activation.pop('target_activation')[-1, :].squeeze(0).cpu()) 
        print(f"✓ Extracted concept vector for '{word}' ({i+1}/{len(concept_words)})", flush = True)


    return activations_list, concept_words

def get_mean_vector(baseline_words, model, tokenizer, layer_idx):
    baseline_activations, _ = extract_concept_vectors(
        model=model,
        tokenizer=tokenizer,
        concept_words=baseline_words,
        layer_idx=layer_idx,
        batch_size=1,
    )
    stacked = torch.stack(baseline_activations)
    mean_vector = stacked.mean(dim=0)
    return mean_vector
def mean_center_vectors(activations_list, model, tokenizer, baseline_words, layer_idx):
    """Mean center a list of vectors"""
    print("Mean centering", flush = True)
    mean_vector = get_mean_vector(
        baseline_words=baseline_words,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
    )
    print("✓ Computed mean vector for mean-centering", flush = True)
    centered = [vec - mean_vector for vec in activations_list]
    print("✓ Mean-centered all concept vectors", flush = True)
    return centered






# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def save_concept_vectors(concept_vectors, concept_labels, filepath):
    """Save concept vectors to disk"""
    torch.save({
        'vectors': torch.stack(concept_vectors),
        'labels': concept_labels,
    }, filepath)
    print(f"✓ Saved concept vectors to {filepath}")


def load_concept_vectors(filepath):
    """Load concept vectors from disk"""
    data = torch.load(filepath)
    vectors = [data['vectors'][i] for i in range(len(data['vectors']))]
    labels = data['labels']
    return vectors, labels


def main():
    try:
        model, tokenizer = get_model()
        
        # Determine layer (about 2/3 through the model)
        num_layers = len(model.model.layers)
        layer_idx = int(num_layers * 0.66)
        print(f"Using layer {layer_idx} out of {num_layers}")
        
        baseline_words = [
            "desks", "jackets", "gondolas", "laughter", "intelligence", "bicycles",
            "chairs", "orchestras", "sand", "pottery", "arrowheads", "jewelry",
            "daffodils", "plateaus", "estuaries", "quilts", "moments", "bamboo",
            "ravines", "archives", "hieroglyphs", "stars", "clay", "fossils",
            "wildlife", "flour", "traffic", "bubbles", "honey", "geodes", "magnets",
            "ribbons", "zigzags", "puzzles", "tornadoes", "anthills", "galaxies",
            "poverty", "diamonds", "universes", "vinegar", "nebulae", "knowledge",
            "marble", "fog", "rivers", "scrolls", "silhouettes", "marbles", "cakes",
            "valleys", "whispers", "pendulums", "towers", "tables", "glaciers",
            "whirlpools", "jungles", "wool", "anger", "ramparts", "flowers",
            "research", "hammers", "clouds", "justice", "dogs", "butterflies",
            "needles", "fortresses", "bonfires", "skyscrapers", "caravans",
            "patience", "bacon", "velocities", "smoke", "electricity", "sunsets",
            "anchors", "parchments", "courage", "statues", "oxygen", "time",
            "butterflies", "fabric", "pasta", "snowflakes", "mountains", "echoes",
            "pianos", "sanctuaries", "abysses", "air", "dewdrops", "gardens",
            "literature", "rice", "enigmas"
        ]

        concept_words = [
            # From the paper's list
            "dust", "satellites", "trumpets", "origami", "illusions",
            "cameras", "lightning", "constellations", "treasures", "phones",
            "trees", "avalanches", "mirrors", "fountains", "quarries",
            "sadness", "xylophones", "secrecy", "oceans", "information",
            "deserts", "kaleidoscopes", "sugar", "vegetables", "poetry",
            "aquariums", "bags", "peace", "caverns", "memories",
            "frosts", "volcanoes", "boulders", "harmonies", "masquerades",
            "rubber", "plastic", "blood", "amphitheaters", "contraptions",
            "youths", "dynasties", "snow", "dirigibles", "algorithms",
            "denim", "monoliths", "milk", "bread", "silver", "something"
        ]
        
        vectors, labels = extract_concept_vectors(
            model=model,
            tokenizer=tokenizer,
            concept_words=concept_words,
            layer_idx=layer_idx,
            batch_size=1,
        )
        print("✓ Extracted all concept vectors")
        vectors = mean_center_vectors(vectors, model, tokenizer, baseline_words, layer_idx)
        print("✓ Mean-centered all concept vectors")


        
        # Save for later use
        save_concept_vectors(vectors, labels, "concept_vectors.pt")
        
        
        
        print("\nVector statistics:")
        print(f"Number of vectors: {len(vectors)}")
        print(f"Vector dimension: {vectors[0].shape[0]}")
        print(f"Mean vector norm: {torch.stack([v.norm() for v in vectors]).mean():.2f}")
        print(f"Std vector norm: {torch.stack([v.norm() for v in vectors]).std():.2f}")
        
        # Check some example similarities
        print("\nExample vector similarities:")
        for i in range(min(3, len(vectors))):
            for j in range(i+1, min(3, len(vectors))):
                sim = torch.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
                print(f"  {labels[i]} <-> {labels[j]}: {sim.item():.3f}")
    finally:
        model.model.layers[layer_idx].register_forward_hook(None)
if __name__ == "__main__":
    main()