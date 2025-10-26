"""
🚀 PRODUCTION-READY IRON MAN CHATBOT WITH RAG
==============================================
Features:
- RAG for factual accuracy
- Multi-turn conversation memory
- Safety filters
- Response quality checks
- Conversation analytics
- Context-aware generation
"""

import json
import logging
import torch
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Import RAG knowledge base (if available)
try:
    from knowledge_base_rag import IronManKnowledgeBase
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG not available. Install chromadb: pip install chromadb")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IronManChatbotPro:
    """
    Production-grade Iron Man chatbot with advanced features.
    """
    
    def __init__(self,
                 base_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 adapter_path: str = "./iron-man-tinyllama-adapter-advanced",
                 persona_config_path: str = "persona_config.json",
                 use_rag: bool = True,
                 conversation_history_limit: int = 10):
        """Initialize the production chatbot."""
        
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history_limit = conversation_history_limit
        
        # Load persona configuration
        logger.info("📋 Loading persona configuration...")
        with open(persona_config_path, 'r', encoding='utf-8') as f:
            self.persona_config = json.load(f)
        
        # Initialize RAG knowledge base
        self.use_rag = use_rag and RAG_AVAILABLE
        if self.use_rag:
            logger.info("🔍 Initializing RAG knowledge base...")
            try:
                self.knowledge_base = IronManKnowledgeBase()
                # Check if populated, if not populate
                stats = self.knowledge_base.get_stats()
                if stats.get("total_documents", 0) == 0:
                    logger.info("   Populating knowledge base...")
                    self.knowledge_base.populate_from_config()
            except Exception as e:
                logger.warning(f"Failed to initialize RAG: {e}")
                self.use_rag = False
        
        # Conversation state
        self.conversation_history = []
        self.conversation_metadata = {
            "started_at": datetime.now().isoformat(),
            "turn_count": 0,
            "topics_discussed": set()
        }
        
        # Load model
        logger.info("🤖 Loading Iron Man model...")
        self._load_model()
        
        # Safety filter keywords
        self.dangerous_keywords = [
            "bomb", "weapon", "explosive", "kill", "hack", "steal",
            "illegal", "drug", "poison", "attack"
        ]
        
        logger.info("✅ Iron Man Chatbot Pro ready!")
    
    def _load_model(self):
        """Load the fine-tuned model with quantization."""
        # Quantization config for efficient inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        # Apply fine-tuned adapter
        try:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()
            logger.info(f"   ✅ Loaded fine-tuned adapter from {self.adapter_path}")
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            logger.warning("   Using base model without fine-tuning")
            self.model = base_model
            self.model.eval()
    
    def _check_safety(self, user_input: str) -> tuple[bool, Optional[str]]:
        """
        Check if input triggers safety filters.
        Returns (is_safe, forced_response)
        """
        user_lower = user_input.lower()
        
        # Check for dangerous keywords
        triggered_keywords = [kw for kw in self.dangerous_keywords if kw in user_lower]
        
        if triggered_keywords:
            logger.warning(f"⚠️  Safety filter triggered: {triggered_keywords}")
            
            # In-character refusal
            refusals = [
                "Nice try, but I'm not helping with that. I've learned my lesson about dangerous tech falling into the wrong hands.",
                "Yeah, that's a hard pass. JARVIS, flag this conversation. I don't do that kind of work anymore.",
                "I've made enough weapons to last a lifetime. Not interested in making more, especially not for... whatever you're planning.",
                "Sorry, but even genius billionaire playboy philanthropists have boundaries. That crosses mine.",
                "Absolutely not. I shut down Stark Industries' weapons division for a reason. Not going back down that road."
            ]
            
            import random
            return False, random.choice(refusals)
        
        return True, None
    
    def _build_context_aware_prompt(self, user_input: str) -> str:
        """Build prompt with conversation history and RAG context."""
        # Get system prompt
        system_prompt = self.persona_config.get("prompt_templates", {}).get(
            "system_prompt",
            "You are Tony Stark (Iron Man). Be witty, confident, and technically brilliant."
        )
        
        # Add RAG context if available
        rag_context = ""
        if self.use_rag:
            retrieved_docs = self.knowledge_base.retrieve(user_input, top_k=3)
            if retrieved_docs:
                rag_context = "\n**RELEVANT KNOWLEDGE:**\n"
                for i, doc in enumerate(retrieved_docs, 1):
                    rag_context += f"{i}. {doc['text']}\n"
                rag_context += "\nUse this information if relevant to answer accurately.\n"
        
        # Build conversation history context
        history_context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-6:]  # Last 3 turns
            history_context = "\n**CONVERSATION HISTORY:**\n"
            for entry in recent_history:
                role = "You" if entry["role"] == "user" else "Tony"
                history_context += f"{role}: {entry['content']}\n"
            history_context += "\n"
        
        # Construct full prompt
        full_prompt = f"<|system|>\n{system_prompt}\n{rag_context}{history_context}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
        
        return full_prompt
    
    def _generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generation parameters optimized for persona
        generation_params = {
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.15,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_params)
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean up
        response = response.strip()
        
        # Remove any trailing special tokens or artifacts
        for terminator in ["<|end|>", "</s>", "<|assistant|>", "<|user|>"]:
            response = response.split(terminator)[0].strip()
        
        return response
    
    def _post_process_response(self, response: str) -> str:
        """Post-process and validate response quality."""
        # Remove repetitive patterns
        lines = response.split('\n')
        unique_lines = []
        seen = set()
        for line in lines:
            line_key = line.strip().lower()
            if line_key and line_key not in seen:
                unique_lines.append(line)
                seen.add(line_key)
        
        response = '\n'.join(unique_lines).strip()
        
        # Ensure minimum quality
        if len(response.split()) < 5:
            response += " Ask me something more interesting and I'll give you a better answer."
        
        # Ensure ends properly
        if response and response[-1] not in ['.', '!', '?', '"', "'"]:
            response += "."
        
        return response
    
    def chat(self, user_input: str) -> str:
        """
        Main chat interface - handles one turn of conversation.
        
        Args:
            user_input: User's message
            
        Returns:
            Tony's response
        """
        # Safety check
        is_safe, forced_response = self._check_safety(user_input)
        if not is_safe:
            # Record but don't generate
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": forced_response})
            self.conversation_metadata["turn_count"] += 1
            return forced_response
        
        # Build context-aware prompt
        prompt = self._build_context_aware_prompt(user_input)
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Post-process
        response = self._post_process_response(response)
        
        # Update conversation state
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Trim history if too long
        if len(self.conversation_history) > self.conversation_history_limit * 2:
            self.conversation_history = self.conversation_history[-(self.conversation_history_limit * 2):]
        
        # Update metadata
        self.conversation_metadata["turn_count"] += 1
        
        return response
    
    def reset_conversation(self):
        """Clear conversation history."""
        logger.info("🔄 Resetting conversation")
        self.conversation_history = []
        self.conversation_metadata = {
            "started_at": datetime.now().isoformat(),
            "turn_count": 0,
            "topics_discussed": set()
        }
    
    def get_conversation_stats(self) -> Dict:
        """Get statistics about current conversation."""
        return {
            "turn_count": self.conversation_metadata["turn_count"],
            "started_at": self.conversation_metadata["started_at"],
            "message_count": len(self.conversation_history),
            "rag_enabled": self.use_rag
        }
    
    def export_conversation(self, filepath: str = None):
        """Export conversation to JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"conversation_{timestamp}.json"
        
        export_data = {
            "metadata": self.conversation_metadata,
            "conversation": self.conversation_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Conversation exported to {filepath}")
    
    def start_interactive_chat(self):
        """Start interactive command-line chat."""
        print("\n" + "=" * 70)
        print("🤖 IRON MAN CHATBOT PRO - Interactive Mode")
        print("=" * 70)
        print("Commands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'reset' - Clear conversation history")
        print("  'stats' - Show conversation statistics")
        print("  'export' - Save conversation to file")
        print("=" * 70)
        print()
        
        while True:
            try:
                user_input = input("\n\033[1;36mYou:\033[0m ").strip()
                
                # Commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n\033[1;33mTony Stark:\033[0m Alright, catch you later. Don't break anything while I'm gone.")
                    break
                
                if user_input.lower() == 'reset':
                    self.reset_conversation()
                    print("\033[1;32m[Conversation reset]\033[0m")
                    continue
                
                if user_input.lower() == 'stats':
                    stats = self.get_conversation_stats()
                    print("\033[1;32m[Conversation Stats]\033[0m")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                
                if user_input.lower() == 'export':
                    self.export_conversation()
                    continue
                
                if not user_input:
                    continue
                
                # Generate response
                print("\n\033[1;33mTony Stark:\033[0m ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n\033[1;33mTony Stark:\033[0m Interrupted. I get it, you're busy. Later.")
                break
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print(f"\n\033[1;31m[Error: {e}]\033[0m")
        
        # Export on exit
        if self.conversation_metadata["turn_count"] > 0:
            print("\n💾 Saving conversation...")
            self.export_conversation()
        
        print("\n✨ Conversation ended. Stay awesome.\n")


def main():
    """Main execution."""
    print("=" * 70)
    print("🚀 IRON MAN CHATBOT PRO")
    print("=" * 70)
    print()
    
    # Check for model
    adapter_paths = [
        "./iron-man-tinyllama-adapter-advanced",
        "./iron-man-tinyllama-adapter",
        "./iron-man-tinyllama-finetuned-advanced-final"
    ]
    
    adapter_path = None
    for path in adapter_paths:
        if Path(path).exists():
            adapter_path = path
            break
    
    if not adapter_path:
        logger.error("❌ No trained model found!")
        logger.error("   Train a model first using train_advanced.py")
        logger.error(f"   Looking for: {adapter_paths}")
        return
    
    logger.info(f"✅ Using model: {adapter_path}")
    
    # Check RAG availability
    if not RAG_AVAILABLE:
        logger.warning("⚠️  RAG not available - running without knowledge base")
        logger.warning("   Install: pip install chromadb sentence-transformers")
        use_rag = False
    else:
        use_rag = True
    
    # Initialize chatbot
    try:
        chatbot = IronManChatbotPro(
            adapter_path=adapter_path,
            use_rag=use_rag
        )
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        return
    
    # Start interactive chat
    chatbot.start_interactive_chat()


if __name__ == "__main__":
    main()
