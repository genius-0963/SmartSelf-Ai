"""\
SmartShelf AI - Multi-lingual Processor

Language detection and translation support.
"""

import logging
from typing import Dict, Any, Optional

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

logger = logging.getLogger(__name__)


class MultiLingualProcessor:
    """Detect language and translate queries when needed."""

    def __init__(self):
        self.lang_detector = None
        self.translator = None

        # Lightweight defaults; may download models at runtime.
        if TRANSFORMERS_AVAILABLE:
            try:
                self.lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
                logger.info("Language detection model loaded")
            except Exception as e:
                logger.warning(f"Failed to load language detector: {e}")

            try:
                self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
                logger.info("Translation model loaded")
            except Exception as e:
                logger.warning(f"Failed to load translator: {e}")

    def detect_language(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"language": "en", "confidence": 0.0}

        if self.lang_detector:
            try:
                out = self.lang_detector(text[:256])[0]
                label = out.get("label", "en")
                score = float(out.get("score", 0.0))
                # Model labels are like 'en', 'es', etc.
                return {"language": label, "confidence": round(score, 3)}
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")

        # Fallback heuristic
        if any(ch in text for ch in ["¿", "¡"]):
            return {"language": "es", "confidence": 0.6}
        return {"language": "en", "confidence": 0.5}

    def translate_to_english(self, text: str, source_language: Optional[str] = None) -> Dict[str, Any]:
        if not text:
            return {"translated_text": "", "source_language": source_language or "unknown"}

        if source_language is None:
            source_language = self.detect_language(text)["language"]

        if source_language == "en":
            return {"translated_text": text, "source_language": "en"}

        if self.translator:
            try:
                out = self.translator(text[:512])[0]
                return {"translated_text": out.get("translation_text", text), "source_language": source_language}
            except Exception as e:
                logger.warning(f"Translation failed: {e}")

        # Fallback: no translation
        return {"translated_text": text, "source_language": source_language}
