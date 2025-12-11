import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# 1. Hugging Face model ID
model_id = "Qwen/Qwen2.5-3B-Instruct"

# 2. Local directory to save the model and tokenizer
local_dir = "./Qwen2.5-3B-Instruct_local"

# 3. Determine the device (use cuda if available, otherwise cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------
## SECTION 1: Download and Save Locally
# ----------------------------------------------------

print(f"1. Downloading model and tokenizer from: {model_id}")

# The from_pretrained() method downloads the model/tokenizer files 
# and caches them, but we use save_pretrained() to copy them 
# explicitly to our specified local directory.
try:
    # Load Model (Downloads if not in cache)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto", # Automatically selects the best dtype (e.g., bfloat16 or float16)
        device_map="auto"   # Automatically distributes the model across available devices
    )
    
    # Load Tokenizer (Downloads if not in cache)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"2. Saving model and tokenizer to: {local_dir}")
    # Save the model and tokenizer to the local directory
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

    print("--- Save Complete! ---")

except Exception as e:
    print(f"An error occurred during download/save: {e}")


