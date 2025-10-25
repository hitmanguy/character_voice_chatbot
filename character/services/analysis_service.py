import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
from config import Config
import logging

class TextAnalysisService:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        logging.info("Loading emotion detection model to CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.emotion_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.emotion_model_name).to("cpu")
        logging.info("✅ Emotion model loaded.")
        
    def detect_emotion(self, text: str) -> dict:
        logging.info(f"Detecting emotion for: '{text}'")
        model_on_device = self.model.to(self.device)
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = model_on_device(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            emotion_idx = torch.argmax(probs, dim=-1).item()
            label = model_on_device.config.id2label[emotion_idx]
            logging.info(f"Detected emotion: {label}")
            return {"label": label, "confidence": probs[0][emotion_idx].item()}
        finally:
            model_on_device.to("cpu")