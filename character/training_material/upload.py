import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import getpass # To hide the token when you paste it

# --- 1. Configuration ---

# The base model you used for fine-tuning
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# The path to your saved LoRA adapter
adapter_path = "./iron-man-tinyllama-adapter-advanced"

# The name of the new repository on the Hugging Face Hub
# IMPORTANT: Replace "YourUsername" with your actual Hugging Face username.
# This will be the public name of your model.
hub_repo_id = "Hitmanguy/tinyllama-ironman_vfinal"

# --- 2. Load Models and Tokenizer ---

print(f"Loading base model: {base_model_id}")
# We load the base model in full precision to ensure a clean merge
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
)

print(f"Loading LoRA adapter from: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# --- 3. Merge the Adapter into the Base Model ---

print("Merging the adapter into the base model...")
# This is the key step that creates a new, standalone model
model = model.merge_and_unload()
print("Merge complete.")

# --- 4. Push the Merged Model and Tokenizer to the Hub ---

print(f"\nUploading the merged model to the Hugging Face Hub at: {hub_repo_id}")
print("You will be prompted for your Hugging Face access token.")

try:
    # Use the more secure getpass to hide the token input
    hf_token = getpass.getpass("Enter your Hugging Face token (with write permissions): ")
    
    # Push the model to the Hub
    model.push_to_hub(hub_repo_id, token=hf_token)
    
    # Push the tokenizer to the Hub
    tokenizer.push_to_hub(hub_repo_id, use_auth_token=hf_token)
    
    print("\n✅ Upload complete!")
    print(f"You can now use your model from anywhere with the ID: '{hub_repo_id}'")

except Exception as e:
    print(f"\n❌ An error occurred during upload: {e}")
    print("Please ensure your token has 'write' permissions and the repo name is correct.")