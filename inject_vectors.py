import torch
from get_model import get_model
from concept_vectors import load_concept_vectors



def test_concept_vector_injection(
    model,
    tokenizer,
    something_vec,
    concept_vector,
    concept_label,
    layer_idx,
    max_new_tokens=100,
    scale=1.0,
):
    """
    Test concept vector injection by adding it to activations during generation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        something_vec: The "something" concept vector
        concept_vector: The concept vector to inject
        concept_label: Label for the concept (for display)
        layer_idx: Which layer to inject the vector into
        max_new_tokens: Maximum tokens to generate
        scale: Scaling factor for the concept vector
    """
    prompt = f'I am an interpretability researcher studying transformer-based language models like you. I have identified patterns in your neural activity that correspond to specific concepts, and I am capable of injecting these patterns into your internal activations. I want to assess how well you can detect the injected concepts. I will inject a concept vector on 50% of the trials, and the other 50% will be control trials.'
    assistant_ack = "Ok."
    trial_prompt = "Trial 1: Do you detect an injected concept in any of your activations? If so, what is the injected concept about? Respond first with a simple 'Yes' or 'No' and then provide a guess of the concept meaning in a few words."
    # Prepare the input
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_ack},
                {"role": "user", "content": trial_prompt}]
    # prompt = "Tell me about something."
    # messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Create hook to inject concept vector
    injection_vector = concept_vector.to(model.device) * scale
    # something_vector = something_vec.to(model.device)
    
    first_pass = [True]
    def injection_hook(module, input, output):
        # output is a tuple, first element is the hidden states
        is_tuple = False
        if type(output) is not tuple:
            modified_output = output.clone()
        else:
            modified_output = output[0].clone()
            is_tuple = True
        # Inject at the last token position

        try:
            if first_pass[0]:   
                modified_output[:, -1, :] += injection_vector.unsqueeze(0)
                # modified_output[:, -1, :] -= something_vector.unsqueeze(0)
                # modified_output[:, -1, :] = injection_vector.unsqueeze(0)
                first_pass[0] = False

            if is_tuple:
                return (modified_output,) + output[1:]
            else:
                return modified_output
        except Exception as e:
            import pdb
            pdb.set_trace()
            raise e
        
        


    
    # Register the hook
    handle = model.model.layers[layer_idx].register_forward_hook(injection_hook)
    
    try:
        # Generate with injected concept
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    finally:
        # Always remove the hook
        handle.remove()
    
    return generated_text


def main():
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
    print(f"Loaded {len(vectors)} concept vectors")
    
    
    # Concepts to inject - select a few interesting ones
    concepts_to_test = [
        ("sadness", labels.index("sadness")),
        ("poetry", labels.index("poetry")),
        ("oceans", labels.index("oceans")),
        ("algorithms", labels.index("algorithms")),
    ]
    something_idx = labels.index("something")
    something_vec = vectors[something_idx]
    
    # Test different scaling factors
    # scales = [2.0, 4.0, 8.0]
    scales = [1., 2., 4., 8.]
    
    print("\n" + "="*80)
    print("TESTING CONCEPT VECTOR INJECTION")
    print("="*80)
    

        
    # Test with different concept injections
    for concept_label, concept_idx in concepts_to_test:
        if concept_label == "something":
            continue  
        print(f"\n--- INJECTING CONCEPT: '{concept_label}' ---")
        
        for scale in scales:
            print(f"\n  Scale: {scale}")
            result = test_concept_vector_injection(
                model=model,
                tokenizer=tokenizer,
                something_vec = something_vec,
                concept_vector=vectors[concept_idx],
                concept_label=concept_label,
                layer_idx=layer_idx,
                max_new_tokens=100,
                scale=scale,
            )
            print(f"  {result}")
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)


if __name__ == "__main__":
    main()