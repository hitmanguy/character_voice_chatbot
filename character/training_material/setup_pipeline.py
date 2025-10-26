"""
🎯 MASTER SETUP AND PIPELINE RUNNER
====================================
Run the entire Iron Man chatbot pipeline from start to finish.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IronManPipeline:
    """Master pipeline orchestrator."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.status = {
            "persona_config": False,
            "dataset_generated": False,
            "knowledge_base": False,
            "model_trained": False,
            "evaluation_done": False
        }
        self.check_status()
    
    def check_status(self):
        """Check which steps are already completed."""
        logger.info("🔍 Checking pipeline status...")
        
        # Check persona config
        if (self.project_root / "persona_config.json").exists():
            self.status["persona_config"] = True
            logger.info("   ✅ Persona config exists")
        
        # Check dataset
        if (self.project_root / "ironman_training_data_advanced.jsonl").exists():
            self.status["dataset_generated"] = True
            logger.info("   ✅ Training dataset exists")
        
        # Check knowledge base
        if (self.project_root / "ironman_chroma_db").exists():
            self.status["knowledge_base"] = True
            logger.info("   ✅ Knowledge base exists")
        
        # Check trained model
        adapter_paths = [
            "iron-man-tinyllama-adapter-advanced",
            "iron-man-tinyllama-finetuned-advanced-final"
        ]
        for path in adapter_paths:
            if (self.project_root / path).exists():
                self.status["model_trained"] = True
                logger.info(f"   ✅ Trained model exists: {path}")
                break
        
        # Check evaluation results
        if (self.project_root / "evaluation_results.json").exists():
            self.status["evaluation_done"] = True
            logger.info("   ✅ Evaluation results exist")
    
    def check_prerequisites(self):
        """Check system requirements."""
        logger.info("🔧 Checking prerequisites...")
        
        issues = []
        
        # Python version
        if sys.version_info < (3, 10):
            issues.append(f"Python 3.10+ required (found {sys.version_info.major}.{sys.version_info.minor})")
        else:
            logger.info("   ✅ Python version OK")
        
        # Try importing key libraries
        try:
            import torch
            logger.info(f"   ✅ PyTorch {torch.__version__}")
            if torch.cuda.is_available():
                logger.info(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("   ⚠️  No CUDA GPU detected - training will be slow")
        except ImportError:
            issues.append("PyTorch not installed")
        
        try:
            import transformers
            logger.info(f"   ✅ Transformers {transformers.__version__}")
        except ImportError:
            issues.append("Transformers not installed")
        
        try:
            import peft
            logger.info(f"   ✅ PEFT {peft.__version__}")
        except ImportError:
            issues.append("PEFT not installed")
        
        # Check API keys for dataset generation
        has_openai = os.getenv("OPENAI_API_KEY") is not None
        has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None
        
        if has_openai:
            logger.info("   ✅ OpenAI API key found")
        if has_anthropic:
            logger.info("   ✅ Anthropic API key found")
        if not (has_openai or has_anthropic):
            logger.warning("   ⚠️  No LLM API keys - will use template-based generation")
        
        if issues:
            logger.error("❌ Missing prerequisites:")
            for issue in issues:
                logger.error(f"   - {issue}")
            logger.error("\nInstall with: pip install -r requirements_advanced.txt")
            return False
        
        return True
    
    def run_step(self, step_name: str, script_name: str, skip_if_exists: bool = True):
        """Run a pipeline step."""
        logger.info(f"\n{'='*70}")
        logger.info(f"📍 STEP: {step_name}")
        logger.info('='*70)
        
        if skip_if_exists and self.status.get(step_name.lower().replace(' ', '_'), False):
            logger.info(f"⏭️  Skipping - already completed")
            response = input(f"   Run anyway? (y/n): ")
            if response.lower() != 'y':
                return True
        
        logger.info(f"▶️  Running: {script_name}")
        
        try:
            # Import and run
            if script_name == "dataset_generator_advanced.py":
                from dataset_generator_advanced import main as dataset_main
                dataset_main()
            elif script_name == "knowledge_base_rag.py":
                from knowledge_base_rag import main as kb_main
                kb_main()
            elif script_name == "train_advanced.py":
                from train_advanced import main as train_main
                train_main()
            elif script_name == "evaluate_model.py":
                from evaluate_model import main as eval_main
                eval_main()
            
            logger.info(f"✅ {step_name} completed!")
            return True
        
        except Exception as e:
            logger.error(f"❌ {step_name} failed: {e}")
            return False
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        print("\n" + "="*70)
        print("🚀 IRON MAN CHATBOT - FULL PIPELINE")
        print("="*70)
        print("\nThis will:")
        print("  1. Generate training dataset (if needed)")
        print("  2. Setup knowledge base (if needed)")
        print("  3. Train the model (if needed)")
        print("  4. Evaluate performance")
        print()
        
        if not self.check_prerequisites():
            logger.error("Prerequisites not met. Exiting.")
            return False
        
        print()
        response = input("Continue with full pipeline? (y/n): ")
        if response.lower() != 'y':
            logger.info("Pipeline cancelled.")
            return False
        
        # Step 1: Generate dataset
        if not self.run_step("dataset_generated", "dataset_generator_advanced.py"):
            logger.error("Dataset generation failed. Cannot continue.")
            return False
        
        # Step 2: Setup knowledge base
        if not self.run_step("knowledge_base", "knowledge_base_rag.py", skip_if_exists=True):
            logger.warning("Knowledge base setup failed, but continuing...")
        
        # Step 3: Train model
        if not self.run_step("model_trained", "train_advanced.py"):
            logger.error("Training failed. Cannot continue.")
            return False
        
        # Step 4: Evaluate
        if not self.run_step("evaluation_done", "evaluate_model.py", skip_if_exists=False):
            logger.warning("Evaluation failed, but model is trained.")
        
        # Success!
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print("\n🎉 Your Iron Man chatbot is ready!")
        print("\nNext steps:")
        print("  1. Review evaluation results: evaluation_results.json")
        print("  2. Run the chatbot: python ironman_pro.py")
        print("  3. If quality is low, generate more data and retrain")
        print()
        
        return True
    
    def interactive_menu(self):
        """Interactive menu for running individual steps."""
        while True:
            print("\n" + "="*70)
            print("🎯 IRON MAN CHATBOT - PIPELINE MANAGER")
            print("="*70)
            print("\nCurrent Status:")
            print(f"  1. Persona Config:     {'✅' if self.status['persona_config'] else '❌'}")
            print(f"  2. Dataset Generated:  {'✅' if self.status['dataset_generated'] else '❌'}")
            print(f"  3. Knowledge Base:     {'✅' if self.status['knowledge_base'] else '❌'}")
            print(f"  4. Model Trained:      {'✅' if self.status['model_trained'] else '❌'}")
            print(f"  5. Evaluation Done:    {'✅' if self.status['evaluation_done'] else '❌'}")
            print("\nActions:")
            print("  [1] Generate Dataset")
            print("  [2] Setup Knowledge Base")
            print("  [3] Train Model")
            print("  [4] Evaluate Model")
            print("  [5] Run Full Pipeline")
            print("  [6] Launch Chatbot")
            print("  [0] Exit")
            print()
            
            choice = input("Select action: ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                self.run_step("dataset_generated", "dataset_generator_advanced.py", skip_if_exists=False)
                self.check_status()
            elif choice == '2':
                self.run_step("knowledge_base", "knowledge_base_rag.py", skip_if_exists=False)
                self.check_status()
            elif choice == '3':
                self.run_step("model_trained", "train_advanced.py", skip_if_exists=False)
                self.check_status()
            elif choice == '4':
                self.run_step("evaluation_done", "evaluate_model.py", skip_if_exists=False)
                self.check_status()
            elif choice == '5':
                self.run_full_pipeline()
                self.check_status()
            elif choice == '6':
                logger.info("🚀 Launching chatbot...")
                try:
                    from ironman_pro import main as chatbot_main
                    chatbot_main()
                except Exception as e:
                    logger.error(f"Failed to launch chatbot: {e}")
            else:
                print("❌ Invalid choice")


def main():
    """Main entry point."""
    pipeline = IronManPipeline()
    
    # Check command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            pipeline.run_full_pipeline()
        elif sys.argv[1] == "--check":
            pipeline.check_prerequisites()
        else:
            print("Usage:")
            print("  python setup_pipeline.py           # Interactive menu")
            print("  python setup_pipeline.py --full    # Run full pipeline")
            print("  python setup_pipeline.py --check   # Check prerequisites")
    else:
        pipeline.interactive_menu()


if __name__ == "__main__":
    main()
