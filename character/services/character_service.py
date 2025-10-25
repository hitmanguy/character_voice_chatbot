import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import Config
import logging

class CharacterResponseService:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.chat_history = []
        self._load_model()

    def _load_model(self):
        """
        MODIFIED: Loads the fully-merged character model directly from the
        Hugging Face Hub using the specified character_model_id.
        """
        logging.info(f"Loading character model from Hub: {self.config.character_model_id}")
        
        # The BNB config is still needed to load the model in 4-bit precision.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.config.torch_dtype
        )
        
        # Load the tokenizer from the Hub repo
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.character_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model directly from the Hub repo with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.character_model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.model.eval()
        logging.info("✅ Character model loaded successfully from Hub.")

    def _build_prompt(self, user_input: str) -> str:
        # This method remains unchanged as its logic is independent of model loading.
        recent_history = self.chat_history[-6:]
        history_str = "\n".join([f"{'You' if e['role'] == 'user' else 'Tony Stark'}: {e['content']}" for e in recent_history])
        return f"""<s>[INST]
**SYSTEM PROMPT:**
Your persona is defined as: '{self.config.persona_description}'
Always stay in character.

**RECENT CONVERSATION:**
{history_str}

**CURRENT QUESTION:**
You: {user_input}
[/INST]
Tony Stark: """

    def generate_response(self, user_input: str) -> str:
        # This method also remains unchanged.
        self.chat_history.append({"role": "user", "content": user_input})
        prompt = self._build_prompt(user_input)
        
        generation_params = {
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 0.70,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_params)

        response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        final_response = response_text.split("[/INST]")[0].strip()
        
        self.chat_history.append({"role": "assistant", "content": final_response})
        return final_response