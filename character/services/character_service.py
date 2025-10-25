import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import Config
import logging

# <-- NEW: Import the Gemini LLM function for summarization
from llm_integration import generate_llm_response

class CharacterResponseService:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.chat_history = []
        # <-- NEW: Define a threshold for when to start summarizing
        self.SUMMARY_THRESHOLD = 6  # Start summarizing after 3 conversational turns
        self._load_model()

    def _load_model(self):
        """
        Loads the character model (can be from Hub or local adapter based on your setup).
        This example assumes the Hub-based loading from your last request.
        """
        logging.info(f"Loading character model from Hub: {self.config.character_model_id}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.config.torch_dtype
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.character_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.character_model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.model.eval()
        logging.info("✅ Character model loaded successfully from Hub.")

    def _summarize_history(self) -> str:
        """
        <-- NEW METHOD -->
        Uses a powerful LLM (Gemini) to create a concise summary of the recent conversation
        from the perspective of Tony Stark.
        """
        logging.info("Chat history is long. Generating a summary with Gemini...")
        
        # Format a significant portion of the history for the summarizer
        history_to_summarize = self.chat_history[-10:] # Summarize the last 5 turns
        history_text = "\n".join([f"{'User' if entry['role'] == 'user' else 'Tony Stark'}: {entry['content']}" for entry in history_to_summarize])

        # A professional, persona-aware summarization prompt
        prompt = f"""
        You are a summarization AI. Your task is to read the following conversation and produce a very brief, third-person summary from the perspective of the character 'Tony Stark'.

        Focus on the key topics discussed and the user's main questions.

        CONVERSATION:
        ---
        {history_text}
        ---

        Brief summary from Tony Stark's perspective:
        """
        
        summary = generate_llm_response(prompt)
        logging.info(f"Generated summary: '{summary}'")
        return summary

    def _build_prompt(self, user_input: str) -> str:
        """
        <-- MODIFIED METHOD -->
        Builds a dynamic prompt. If the conversation is long, it uses a generated summary.
        Otherwise, it uses the raw recent chat history.
        """
        history_context = ""
        prompt_header = ""

        # Decide whether to summarize or use raw history
        if len(self.chat_history) > self.SUMMARY_THRESHOLD:
            history_context = self._summarize_history()
            prompt_header = "SUMMARY OF RECENT CONVERSATION:"
        else:
            recent_history = self.chat_history[-6:]
            history_context = "\n".join([f"{'You' if e['role'] == 'user' else 'Tony Stark'}: {e['content']}" for e in recent_history])
            prompt_header = "RECENT CONVERSATION:"

        return f"""<s>[INST]
**SYSTEM PROMPT:**
Your persona is defined as: '{self.config.persona_description}'
Always stay in character.

**{prompt_header}**
{history_context}

**CURRENT QUESTION:**
You: {user_input}
[/INST]
Tony Stark: """

    def generate_response(self, user_input: str) -> str:
        # This method remains the same; the complexity is handled by _build_prompt.
        self.chat_history.append({"role": "user", "content": user_input})
        prompt = self._build_prompt(user_input)
        
        generation_params = {
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": 0.70,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_params)

        response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        final_response = response_text.split("[/INST]")[0].strip()
        
        self.chat_history.append({"role": "assistant", "content": final_response})
        return final_response