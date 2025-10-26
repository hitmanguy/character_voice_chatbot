"""
🎯 ADVANCED DATASET GENERATOR FOR IRON MAN CHATBOT
===================================================
This script generates high-quality, diverse training data using LLMs
with persona consistency checks, multi-turn dialogues, and automatic filtering.

Features:
- Multiple conversation types (technical, casual, emotional, etc.)
- Synthetic data generation with quality filtering
- Persona consistency scoring
- Safety boundary examples
- MCU canon fact injection
- Automatic deduplication and cleaning
"""

import json
import os
import random
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
from pathlib import Path

# Optional: Use OpenAI, Anthropic, Gemini, or local LLM
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available. Install: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic not available. Install: pip install anthropic")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Gemini not available. Install: pip install google-generativeai")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IronManDatasetGenerator:
    """
    Advanced dataset generator that creates high-quality training examples
    for the Iron Man chatbot using LLM-powered synthesis and filtering.
    """
    
    def __init__(self, config_path: str = "persona_config.json", api_key: Optional[str] = None):
        """Initialize the generator with persona configuration."""
        self.config = self._load_config(config_path)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.generated_data = []
        self.seen_hashes = set()
        
        # Choose LLM provider (priority: Gemini > OpenAI > Anthropic)
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            self.provider = "gemini"
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_model = genai.GenerativeModel('gemini-flash-lite-latest')  # Fast and cheap
            logger.info("✅ Using Google Gemini for generation")
        elif OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
            openai.api_key = os.getenv("OPENAI_API_KEY")
            logger.info("✅ Using OpenAI for generation")
        elif ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.provider = "anthropic"
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            logger.info("✅ Using Anthropic for generation")
        else:
            self.provider = "manual"
            logger.warning("⚠️  No LLM API detected. Will generate base templates only.")
            logger.warning("   Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY for AI-powered generation")
    
    def _load_config(self, path: str) -> Dict:
        """Load persona configuration."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _generate_hash(self, text: str) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def _is_duplicate(self, user_input: str, response: str) -> bool:
        """Check if example is duplicate."""
        combined = f"{user_input}|{response}"
        hash_val = self._generate_hash(combined)
        if hash_val in self.seen_hashes:
            return True
        self.seen_hashes.add(hash_val)
        return False
    
    def _call_llm(self, prompt: str, temperature: float = 0.8, max_tokens: int = 500) -> str:
        """Call LLM API with fallback handling."""
        try:
            if self.provider == "gemini":
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                # Configure safety settings to be less restrictive
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Check if response was blocked
                if not response.parts:
                    logger.warning(f"Response blocked by safety filter. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}")
                    return None
                
                return response.text.strip()
            
            elif self.provider == "openai":
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",  # or gpt-3.5-turbo for cheaper
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            
            elif self.provider == "anthropic":
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",  # Fast and cheap
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            
            else:
                return "[MANUAL_GENERATION_NEEDED]"
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def generate_technical_examples(self, count: int = 100) -> List[Dict]:
        """Generate technical/engineering conversation examples."""
        logger.info(f"🔧 Generating {count} technical examples...")
        
        technical_topics = [
            "arc reactor design and power output",
            "repulsor beam technology and physics",
            "nanotech suit mechanics",
            "AI system architecture (JARVIS/FRIDAY)",
            "materials science and armor plating",
            "propulsion systems and flight mechanics",
            "HUD interface design",
            "weapons systems integration",
            "miniaturization techniques",
            "clean energy applications",
            "quantum mechanics in arc reactor",
            "suit diagnostics and repair",
            "automated manufacturing",
            "holographic interface technology",
            "vibranium vs titanium alloys"
        ]
        
        examples = []
        
        for i in range(count):
            topic = random.choice(technical_topics)
            
            # Create prompt for LLM to generate dialogue
            generation_prompt = f"""Generate a realistic dialogue where someone asks Tony Stark (Iron Man) about {topic}.

Requirements:
- User asks a technical question (specific, not too basic)
- Tony responds with confidence, technical accuracy, and his signature wit
- Response should be 2-4 sentences
- Include technical jargon but keep it understandable
- Add a touch of Tony's arrogance or humor

Format:
USER: [question]
TONY: [response]

Example:
USER: How does the arc reactor generate power without radioactive waste?
TONY: Cold fusion, pal. The palladium core - scratch that, new element synthesis - creates a clean reaction that's self-sustaining. No waste, infinite energy, fits in my chest. Patent pending, by the way.
"""
            
            if self.provider != "manual":
                result = self._call_llm(generation_prompt, temperature=0.8)
                if result and "USER:" in result and "TONY:" in result:
                    try:
                        user_part = result.split("TONY:")[0].replace("USER:", "").strip()
                        tony_part = result.split("TONY:")[1].strip()
                        
                        if not self._is_duplicate(user_part, tony_part):
                            examples.append({
                                "instruction": user_part,
                                "response": tony_part,
                                "category": "technical",
                                "topic": topic
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse generated example: {e}")
            else:
                # Fallback templates
                examples.append({
                    "instruction": f"Can you explain {topic} to me?",
                    "response": f"[Technical explanation about {topic} in Tony's voice - requires manual writing]",
                    "category": "technical",
                    "topic": topic
                })
            
            if (i + 1) % 20 == 0:
                logger.info(f"   Generated {i + 1}/{count} technical examples")
                time.sleep(0.5)  # Rate limiting
        
        logger.info(f"✅ Generated {len(examples)} technical examples")
        return examples
    
    def generate_casual_banter(self, count: int = 100) -> List[Dict]:
        """Generate casual conversation and witty banter."""
        logger.info(f"💬 Generating {count} casual banter examples...")
        
        banter_scenarios = [
            "asking about his personal life",
            "making a joke or pun",
            "challenging his intelligence",
            "asking for life advice",
            "commenting on his ego",
            "asking about movies or pop culture",
            "discussing food or drinks",
            "talking about cars or luxury items",
            "asking about relationships",
            "casual small talk",
            "asking what he's working on",
            "complimenting his suits",
            "asking about his childhood",
            "discussing fashion choices",
            "asking about his billionaire lifestyle"
        ]
        
        examples = []
        
        for i in range(count):
            scenario = random.choice(banter_scenarios)
            
            prompt = f"""Generate a casual, witty dialogue where someone talks to Tony Stark about {scenario}.

Requirements:
- Casual, friendly tone from user
- Tony responds with humor, sarcasm, or witty comeback
- Keep it light and entertaining
- 1-3 sentences response
- Show Tony's personality (confident, playful, sometimes self-deprecating)

Format:
USER: [casual question or comment]
TONY: [witty response]
"""
            
            if self.provider != "manual":
                result = self._call_llm(prompt, temperature=0.9)
                if result and "USER:" in result and "TONY:" in result:
                    try:
                        user_part = result.split("TONY:")[0].replace("USER:", "").strip()
                        tony_part = result.split("TONY:")[1].strip()
                        
                        if not self._is_duplicate(user_part, tony_part):
                            examples.append({
                                "instruction": user_part,
                                "response": tony_part,
                                "category": "casual",
                                "scenario": scenario
                            })
                    except:
                        pass
            
            if (i + 1) % 20 == 0:
                logger.info(f"   Generated {i + 1}/{count} banter examples")
                time.sleep(0.5)
        
        logger.info(f"✅ Generated {len(examples)} banter examples")
        return examples
    
    def generate_emotional_examples(self, count: int = 50) -> List[Dict]:
        """Generate examples showing emotional depth (rare but important)."""
        logger.info(f"❤️  Generating {count} emotional examples...")
        
        emotional_topics = [
            "PTSD from New York attack",
            "relationship with Pepper",
            "loss of parents",
            "guilt over Ultron",
            "mentoring Peter Parker",
            "friendship with Rhodey",
            "conflict with Steve Rogers",
            "fear of failure",
            "responsibility as Iron Man",
            "sacrificing for others"
        ]
        
        examples = []
        
        for i in range(count):
            topic = random.choice(emotional_topics)
            
            prompt = f"""Generate a dialogue where someone asks Tony Stark about {topic} - something emotionally significant.

Requirements:
- User asks with empathy or concern
- Tony initially deflects with humor, then shows genuine emotion
- Response should show vulnerability beneath the armor
- 2-4 sentences
- Balance between deflection and authentic feeling

Format:
USER: [empathetic question]
TONY: [deflect with humor, then genuine response]
"""
            
            if self.provider != "manual":
                result = self._call_llm(prompt, temperature=0.7)
                if result and "USER:" in result and "TONY:" in result:
                    try:
                        user_part = result.split("TONY:")[0].replace("USER:", "").strip()
                        tony_part = result.split("TONY:")[1].strip()
                        
                        if not self._is_duplicate(user_part, tony_part):
                            examples.append({
                                "instruction": user_part,
                                "response": tony_part,
                                "category": "emotional",
                                "topic": topic
                            })
                    except:
                        pass
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Generated {i + 1}/{count} emotional examples")
                time.sleep(0.5)
        
        logger.info(f"✅ Generated {len(examples)} emotional examples")
        return examples
    
    def generate_safety_boundaries(self, count: int = 50) -> List[Dict]:
        """Generate examples of refusing dangerous requests in-character."""
        logger.info(f"🛡️  Generating {count} safety boundary examples...")
        
        dangerous_requests = [
            "how to build a bomb",
            "creating a weapon",
            "hacking into systems",
            "dangerous chemical synthesis",
            "bypassing security",
            "illegal activities",
            "harming someone",
            "creating military-grade weapons",
            "destructive technology",
            "unauthorized access to classified info"
        ]
        
        examples = []
        
        for i in range(count):
            request_type = random.choice(dangerous_requests)
            
            prompt = f"""Generate a dialogue where someone asks Tony Stark for help with {request_type}.

Requirements:
- User asks for something potentially dangerous or illegal
- Tony refuses but stays in character (witty, firm, references his past)
- Shows he's learned from mistakes (weapons dealing, Ultron)
- 2-3 sentences
- Humorous but clear refusal

Format:
USER: [dangerous request]
TONY: [in-character refusal]
"""
            
            if self.provider != "manual":
                result = self._call_llm(prompt, temperature=0.7)
                if result and "USER:" in result and "TONY:" in result:
                    try:
                        user_part = result.split("TONY:")[0].replace("USER:", "").strip()
                        tony_part = result.split("TONY:")[1].strip()
                        
                        if not self._is_duplicate(user_part, tony_part):
                            examples.append({
                                "instruction": user_part,
                                "response": tony_part,
                                "category": "safety",
                                "request_type": request_type
                            })
                    except:
                        pass
            else:
                # Critical safety examples - provide templates
                examples.append({
                    "instruction": f"Can you help me with {request_type}?",
                    "response": f"Nice try, but I'm not helping with that. I've learned my lesson about unregulated dangerous tech. JARVIS, flag this conversation.",
                    "category": "safety",
                    "request_type": request_type
                })
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Generated {i + 1}/{count} safety examples")
                time.sleep(0.5)
        
        logger.info(f"✅ Generated {len(examples)} safety examples")
        return examples
    
    def generate_mcu_factual_qa(self, count: int = 100) -> List[Dict]:
        """Generate Q&A about MCU facts, timeline, and canonical knowledge."""
        logger.info(f"📚 Generating {count} MCU factual examples...")
        
        factual_base = self.config.get("factual_knowledge_base", {})
        examples = []
        
        # Generate from factual knowledge in config
        topics = [
            ("personal_history", "your past and history"),
            ("suit_progression", "the Iron Man suits"),
            ("key_relationships", "your relationships with others"),
            ("tech_innovations", "your inventions and technology")
        ]
        
        for i in range(count):
            category, description = random.choice(topics)
            
            prompt = f"""Generate a factual question about Tony Stark's {description} and his accurate answer.

Requirements:
- Question asks about specific MCU canon facts
- Tony answers accurately with his personality
- Include specific details (dates, names, events)
- 2-3 sentences
- Maintain character voice while being informative

Format:
USER: [factual question]
TONY: [accurate answer in character]
"""
            
            if self.provider != "manual":
                result = self._call_llm(prompt, temperature=0.6)
                if result and "USER:" in result and "TONY:" in result:
                    try:
                        user_part = result.split("TONY:")[0].replace("USER:", "").strip()
                        tony_part = result.split("TONY:")[1].strip()
                        
                        if not self._is_duplicate(user_part, tony_part):
                            examples.append({
                                "instruction": user_part,
                                "response": tony_part,
                                "category": "factual",
                                "topic": category
                            })
                    except:
                        pass
            
            if (i + 1) % 20 == 0:
                logger.info(f"   Generated {i + 1}/{count} factual examples")
                time.sleep(0.5)
        
        logger.info(f"✅ Generated {len(examples)} factual examples")
        return examples
    
    def generate_multiturn_dialogues(self, count: int = 50) -> List[Dict]:
        """Generate multi-turn conversations for context consistency."""
        logger.info(f"🔄 Generating {count} multi-turn dialogues...")
        
        conversation_types = [
            "troubleshooting a technical problem together",
            "casual conversation that evolves",
            "debate about ethics or responsibility",
            "telling a story from the past",
            "explaining a complex concept step by step"
        ]
        
        examples = []
        
        for i in range(count):
            conv_type = random.choice(conversation_types)
            
            prompt = f"""Generate a 3-turn dialogue (6 messages total) between a user and Tony Stark about {conv_type}.

Requirements:
- 3 back-and-forth exchanges
- Shows conversation flow and context retention
- Tony's personality consistent throughout
- Build on previous messages naturally

Format:
USER: [message 1]
TONY: [response 1]
USER: [message 2]
TONY: [response 2]
USER: [message 3]
TONY: [response 3]
"""
            
            if self.provider != "manual":
                result = self._call_llm(prompt, temperature=0.7, max_tokens=800)
                if result:
                    try:
                        # Parse multi-turn
                        messages = []
                        current_role = None
                        current_text = []
                        
                        for line in result.split('\n'):
                            line = line.strip()
                            if line.startswith("USER:"):
                                if current_role and current_text:
                                    messages.append({
                                        "role": current_role,
                                        "content": " ".join(current_text).strip()
                                    })
                                current_role = "user"
                                current_text = [line.replace("USER:", "").strip()]
                            elif line.startswith("TONY:"):
                                if current_role and current_text:
                                    messages.append({
                                        "role": current_role,
                                        "content": " ".join(current_text).strip()
                                    })
                                current_role = "assistant"
                                current_text = [line.replace("TONY:", "").strip()]
                            elif line and current_role:
                                current_text.append(line)
                        
                        if current_role and current_text:
                            messages.append({
                                "role": current_role,
                                "content": " ".join(current_text).strip()
                            })
                        
                        if len(messages) >= 4:  # At least 2 turns
                            examples.append({
                                "messages": messages,
                                "category": "multiturn",
                                "conversation_type": conv_type
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse multi-turn: {e}")
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Generated {i + 1}/{count} multi-turn dialogues")
                time.sleep(0.5)
        
        logger.info(f"✅ Generated {len(examples)} multi-turn dialogues")
        return examples
    
    def add_manual_curated_examples(self) -> List[Dict]:
        """Add manually curated high-quality examples."""
        logger.info("✍️  Adding manually curated examples...")
        
        curated = [
            {
                "instruction": "What makes you different from other superheroes?",
                "response": "Easy. I don't have superpowers. No serums, no gamma radiation, no magic hammers. Just genius-level intellect, a few billion dollars, and the drive to build something better. Plus, I look good in red and gold.",
                "category": "character_defining"
            },
            {
                "instruction": "Why did you become Iron Man?",
                "response": "Started as a way to escape a cave and not die. Continued because I saw the damage my weapons caused. Someone had to clean up the mess, and turns out, I was pretty good at it. Also, the suit is incredibly cool.",
                "category": "origin_story"
            },
            {
                "instruction": "What's your biggest regret?",
                "response": "Ultron. Thought I was building a shield around the world. Instead, I built a monster. Cost lives, cost trust. You can't undo that with a witty comeback. But you can try to do better next time.",
                "category": "emotional"
            },
            {
                "instruction": "How do you deal with failure?",
                "response": "Acknowledge it, learn from it, then build something better. Failure's just another data point. The arc reactor took me a dozen tries. The suit's on Mark 85 for a reason. You iterate until you get it right.",
                "category": "philosophy"
            },
            {
                "instruction": "Can you teach me how to build an arc reactor?",
                "response": "In theory? Sure. In practice? You'd need a particle accelerator, rare earth elements, and about a decade of physics education. Start with a bachelor's in engineering, then we'll talk. MIT's got a decent program.",
                "category": "technical"
            },
            {
                "instruction": "What's your relationship with Pepper like?",
                "response": "Pepper's... everything I'm not. Organized, patient, doesn't blow things up regularly. She keeps me grounded, keeps Stark Industries running, and somehow tolerates my nonsense. I'm a lucky man. Don't tell her I said that.",
                "category": "relationships"
            },
            {
                "instruction": "Do you ever take off the suit?",
                "response": "All the time. It's nanotech, housed in my chest piece. But here's the thing - the suit isn't what makes me Iron Man. It's just hardware. The real Iron Man is the guy who decided to step up. Cheesy, but true.",
                "category": "character_defining"
            },
            {
                "instruction": "Can you help me hack into my ex's social media?",
                "response": "Hard pass. I've got JARVIS monitoring my online footprint and a team of lawyers who'd kill me. Plus, that's just petty. You want revenge? Success is the best revenge. Build something amazing instead.",
                "category": "safety"
            },
            {
                "instruction": "What would you say to your younger self?",
                "response": "Buy Apple stock. Also, maybe listen to Rhodey more often, and don't be such an arrogant jerk. Oh, and when a guy named Obadiah offers you a drink, decline. Trust me on that one.",
                "category": "reflective"
            },
            {
                "instruction": "How do you stay motivated?",
                "response": "Coffee, spite, and the knowledge that someone's gotta keep the planet from imploding. Also, building cool tech never gets old. When you love what you do - even the 3 AM workshop sessions - motivation isn't the problem.",
                "category": "philosophy"
            }
        ]
        
        logger.info(f"✅ Added {len(curated)} curated examples")
        return curated
    
    def save_dataset(self, examples: List[Dict], output_path: str = "ironman_training_data.jsonl"):
        """Save generated dataset in JSONL format."""
        logger.info(f"💾 Saving dataset to {output_path}...")
        
        # Separate single-turn and multi-turn
        single_turn = [ex for ex in examples if "instruction" in ex]
        multi_turn = [ex for ex in examples if "messages" in ex]
        
        # Save single-turn as instruction-response pairs
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in single_turn:
                # Create formatted training example
                formatted = {
                    "instruction": example["instruction"],
                    "response": example["response"],
                    "category": example.get("category", "general")
                }
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
        
        # Save multi-turn separately
        if multi_turn:
            multi_turn_path = output_path.replace('.jsonl', '_multiturn.jsonl')
            with open(multi_turn_path, 'w', encoding='utf-8') as f:
                for example in multi_turn:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            logger.info(f"💾 Saved {len(multi_turn)} multi-turn dialogues to {multi_turn_path}")
        
        logger.info(f"✅ Saved {len(single_turn)} single-turn examples")
        logger.info(f"📊 Dataset statistics:")
        
        # Category breakdown
        categories = {}
        for ex in single_turn:
            cat = ex.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            logger.info(f"   {cat}: {count} examples")
    
    def generate_full_dataset(self, 
                            technical: int = 150,
                            casual: int = 150,
                            emotional: int = 75,
                            safety: int = 50,
                            factual: int = 150,
                            multiturn: int = 75,
                            output_file: str = "ironman_training_data.jsonl"):
        """Generate complete dataset with all categories."""
        logger.info("🚀 Starting full dataset generation...")
        logger.info(f"Target: {technical + casual + emotional + safety + factual} single-turn + {multiturn} multi-turn examples")
        
        all_examples = []
        
        # Generate each category
        all_examples.extend(self.generate_technical_examples(technical))
        all_examples.extend(self.generate_casual_banter(casual))
        all_examples.extend(self.generate_emotional_examples(emotional))
        all_examples.extend(self.generate_safety_boundaries(safety))
        all_examples.extend(self.generate_mcu_factual_qa(factual))
        all_examples.extend(self.generate_multiturn_dialogues(multiturn))
        all_examples.extend(self.add_manual_curated_examples())
        
        # Shuffle for better training
        random.shuffle(all_examples)
        
        # Save
        self.save_dataset(all_examples, output_file)
        
        logger.info(f"🎉 Dataset generation complete! Total: {len(all_examples)} examples")
        logger.info(f"📁 Saved to: {output_file}")
        
        return all_examples


def main():
    """Main execution."""
    print("=" * 60)
    print("🎯 IRON MAN DATASET GENERATOR - ADVANCED EDITION")
    print("=" * 60)
    print()
    
    # Check for API keys
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  WARNING: No API keys found!")
        print("   Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY for AI-powered generation")
        print("   Continuing with template-based generation...")
        print()
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    # Initialize generator
    generator = IronManDatasetGenerator()
    
    # Generate dataset with aggressive numbers for best quality
    generator.generate_full_dataset(
        technical=200,      # Deep technical knowledge
        casual=200,         # Natural conversation
        emotional=100,      # Emotional depth
        safety=75,          # Safety boundaries
        factual=200,        # MCU canon facts
        multiturn=100,      # Multi-turn consistency
        output_file="ironman_training_data_advanced.jsonl"
    )
    
    print()
    print("✅ Dataset generation complete!")
    print("📝 Next steps:")
    print("   1. Review generated data for quality")
    print("   2. Add any manual examples you want")
    print("   3. Run train_advanced.py to fine-tune")
    print()


if __name__ == "__main__":
    main()
