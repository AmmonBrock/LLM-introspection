from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def get_model():

    local_path = "./Qwen2.5-3B-Instruct_local"   

    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        local_files_only=True,     
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,     
        torch_dtype=torch.bfloat16, 
        device_map="auto",          
        trust_remote_code=True,
        # attn_implementation="eager",  
    )

    model.eval()
    return model, tokenizer

def get_model():
    local_dir = "./Qwen2.5-3B-Instruct_local" 

    # Load the Model from the local directory
    local_model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load the Tokenizer from the local directory
    local_tokenizer = AutoTokenizer.from_pretrained(local_dir)

    print(f"Local Model class: {type(local_model)}")
    print(f"Local Tokenizer class: {type(local_tokenizer)}")
    
    return local_model, local_tokenizer
#     # Optional: A quick test generation to verify the loaded model
#     print("\n4. Running a test inference with the locally loaded model...")
#     prompt = "Write a short poem about a loyal dog."
    
#     messages = [
#         {"role": "user", "content": prompt}
#     ]
    
#     # Apply Qwen's chat template
#     text = local_tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     model_inputs = local_tokenizer([text], return_tensors="pt").to(local_model.device)
    
#     generated_ids = local_model.generate(
#         **model_inputs, 
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.7
#     )
    
#     response = local_tokenizer.decode(
#         generated_ids[0][model_inputs.input_ids.shape[1]:], 
#         skip_special_tokens=True
#     )
    
#     print("\n--- Generated Response ---")
#     print(response.strip())
