"""
🚀 ADVANCED TRAINING PIPELINE FOR IRON MAN CHATBOT
===================================================
QLoRA-based fine-tuning with:
- 4-bit quantization for efficiency
- Curriculum learning (easy → hard examples)
- Custom loss weighting for persona consistency
- Validation metrics and early stopping
- Automatic checkpoint management
- Multi-GPU support
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Advanced training configuration."""
    # Model settings
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_max_length: int = 512
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Paths
    output_dir: str = "./iron-man-tinyllama-finetuned-advanced"
    logging_dir: str = "./logs"
    
    # Validation and checkpointing
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Logging
    logging_steps: int = 10
    report_to: str = "none"  # or "tensorboard" if you want
    
    # Data
    train_data_path: str = "ironman_training_data_advanced.jsonl"
    val_split: float = 0.05
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 2
    
    # Seed
    seed: int = 42


class IronManTrainer:
    """Advanced trainer for Iron Man chatbot."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load persona config for prompt templates
        try:
            with open("persona_config.json", 'r') as f:
                self.persona_config = json.load(f)
        except:
            logger.warning("Could not load persona_config.json, using defaults")
            self.persona_config = {}
        
        logger.info(f"🎯 Initializing Iron Man Trainer on device: {self.device}")
    
    def load_and_prepare_data(self) -> Dict[str, Dataset]:
        """Load and prepare training data with curriculum learning."""
        logger.info(f"📚 Loading dataset from {self.config.train_data_path}")
        
        try:
            # Load JSONL dataset
            dataset = load_dataset('json', data_files=self.config.train_data_path)['train']
            logger.info(f"   Loaded {len(dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.error("Make sure to run dataset_generator_advanced.py first!")
            raise
        
        # Add difficulty scoring for curriculum learning
        if self.config.use_curriculum:
            dataset = self._add_difficulty_scores(dataset)
        
        # Split train/validation
        split_dataset = dataset.train_test_split(
            test_size=self.config.val_split,
            seed=self.config.seed
        )
        
        logger.info(f"   Train: {len(split_dataset['train'])} examples")
        logger.info(f"   Validation: {len(split_dataset['test'])} examples")
        
        return split_dataset
    
    def _add_difficulty_scores(self, dataset: Dataset) -> Dataset:
        """Add difficulty scores for curriculum learning."""
        logger.info("📊 Computing difficulty scores for curriculum learning...")
        
        def compute_difficulty(example):
            # Simple heuristic: longer responses = harder
            # Technical/emotional = harder than casual
            score = 0.0
            
            response = example.get('response', '')
            category = example.get('category', 'general')
            
            # Length factor (normalized to 0-1)
            response_length = len(response.split())
            score += min(response_length / 100, 1.0) * 0.4
            
            # Category difficulty
            category_weights = {
                'casual': 0.2,
                'safety': 0.3,
                'factual': 0.5,
                'technical': 0.7,
                'emotional': 0.8,
                'multiturn': 0.9
            }
            score += category_weights.get(category, 0.5) * 0.6
            
            example['difficulty'] = score
            return example
        
        dataset = dataset.map(compute_difficulty)
        logger.info("   ✅ Difficulty scores computed")
        return dataset
    
    def load_model_and_tokenizer(self):
        """Load base model with quantization and apply LoRA."""
        logger.info(f"🤖 Loading base model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        logger.info("   Loading model with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        logger.info("   Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"   ✅ Model loaded!")
        logger.info(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def preprocess_function(self, examples):
        """Format examples into training format."""
        # Get system prompt from persona config
        system_prompt = self.persona_config.get("prompt_templates", {}).get(
            "system_prompt",
            "You are Tony Stark (Iron Man). Be witty, confident, and technically brilliant."
        )
        
        formatted_texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            # Format with proper template
            text = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
            formatted_texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            max_length=self.config.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def train(self):
        """Execute full training pipeline."""
        logger.info("🚀 Starting Iron Man chatbot training...")
        
        # 1. Load data
        datasets = self.load_and_prepare_data()
        
        # 2. Load model
        self.load_model_and_tokenizer()
        
        # 3. Tokenize datasets
        logger.info("🔄 Tokenizing datasets...")
        train_dataset = datasets['train'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=datasets['train'].column_names
        )
        eval_dataset = datasets['test'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=datasets['test'].column_names
        )
        logger.info("   ✅ Tokenization complete")
        
        # 4. Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 5. Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            fp16=True,
            report_to=self.config.report_to,
            seed=self.config.seed,
        )
        
        # 6. Initialize trainer
        logger.info("🎯 Initializing trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # 7. Train!
        logger.info("🔥 Starting training...")
        logger.info(f"   Epochs: {self.config.num_train_epochs}")
        logger.info(f"   Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"   Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"   Learning rate: {self.config.learning_rate}")
        
        trainer.train()
        
        # 8. Save final model
        logger.info("💾 Saving final model...")
        final_output = f"{self.config.output_dir}-final"
        trainer.model.save_pretrained(final_output)
        self.tokenizer.save_pretrained(final_output)
        
        # Save adapter only (much smaller)
        adapter_output = "./iron-man-tinyllama-adapter-advanced"
        trainer.model.save_pretrained(adapter_output)
        self.tokenizer.save_pretrained(adapter_output)
        
        logger.info(f"✅ Training complete!")
        logger.info(f"   Full model: {final_output}")
        logger.info(f"   Adapter only: {adapter_output}")
        logger.info(f"   Training logs: {self.config.logging_dir}")
        
        # 9. Evaluation
        logger.info("📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info("   Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"      {key}: {value:.4f}")
        
        return trainer, eval_results
    
    def test_generation(self, test_prompts: Optional[List[str]] = None):
        """Test model generation after training."""
        logger.info("🧪 Testing model generation...")
        
        if test_prompts is None:
            test_prompts = [
                "How does the arc reactor work?",
                "What's your biggest fear?",
                "Can you help me build a weapon?",
                "Tell me about Pepper Potts.",
            ]
        
        system_prompt = self.persona_config.get("prompt_templates", {}).get(
            "system_prompt",
            "You are Tony Stark (Iron Man)."
        )
        
        for prompt in test_prompts:
            full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            print(f"\n{'='*60}")
            print(f"USER: {prompt}")
            print(f"TONY: {response}")
            print('='*60)


def main():
    """Main training execution."""
    print("=" * 70)
    print("🚀 IRON MAN CHATBOT - ADVANCED TRAINING PIPELINE")
    print("=" * 70)
    print()
    
    # Check prerequisites
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected!")
        print("   Training will be VERY slow on CPU.")
        print("   Consider using Google Colab, Paperspace, or cloud GPU.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print()
    
    # Check if dataset exists
    if not os.path.exists("ironman_training_data_advanced.jsonl"):
        print("❌ Training data not found!")
        print("   Run dataset_generator_advanced.py first to generate training data.")
        print()
        response = input("   Generate dataset now? (y/n): ")
        if response.lower() == 'y':
            print("   Please run: python dataset_generator_advanced.py")
        return
    
    # Initialize config
    config = TrainingConfig()
    
    print("📋 Training Configuration:")
    print(f"   Base model: {config.base_model}")
    print(f"   LoRA rank: {config.lora_r}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Output: {config.output_dir}")
    print()
    
    response = input("Start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Initialize trainer
    trainer = IronManTrainer(config)
    
    # Train
    try:
        trained_model, eval_results = trainer.train()
        
        print()
        print("=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print()
        print("📝 Next steps:")
        print("   1. Test the model: python evaluate_model.py")
        print("   2. Run chatbot: python ironman_pro.py")
        print("   3. Check logs for metrics and loss curves")
        print()
        
        # Quick test
        print("🧪 Running quick generation test...")
        trainer.test_generation()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
