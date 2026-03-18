"""
Vision analyzer: analisi immagini multi-backend per interpretazione artistica.
PORTED FROM: AgenteIA-Production/src/vision_analysis_engine.py (refactored)
"""

import base64
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """
    Analisi immagini multi-backend con fallback:
    1. Google Gemini Vision (miglior qualità per arte)
    2. DeepSeek-VL2
    3. OCR puro (pytesseract) come fallback
    """

    ANALYSIS_PROMPTS = {
        "quick": "Analizza rapidamente questa immagine: soggetto, stile, colori principali.",
        "standard": (
            "Analizza questa immagine in profondità:\n"
            "1. SOGGETTO: cosa è rappresentato\n"
            "2. STILE: movimento artistico, periodo, influenze\n"
            "3. COMPOSIZIONE: struttura, focal points, equilibrio\n"
            "4. COLORI: palette, contrasti, temperatura\n"
            "5. TECNICHE: pennellate, texture, uso della luce\n"
            "Rispondi in italiano."
        ),
        "deep": (
            "Analisi artistica ESTENSIVA. Rispondi in italiano.\n\n"
            "## ANALISI FORMALE\n"
            "- Composizione e struttura geometrica\n"
            "- Prospettiva e uso dello spazio\n"
            "- Movimento visivo e focal points\n\n"
            "## COLORE E LUCE\n"
            "- Palette dominante e accenti\n"
            "- Contrasti cromatici\n"
            "- Significato simbolico dei colori\n\n"
            "## STILE E TECNICA\n"
            "- Movimento artistico\n"
            "- Tecniche pittoriche\n"
            "- Confronti con artisti noti\n\n"
            "## CONTESTO\n"
            "- Periodo storico stimato\n"
            "- Significato iconografico\n"
            "- Impatto estetico"
        ),
    }

    def __init__(self):
        self.last_metadata: Dict = {}

    def analyze(
        self,
        image_path: str,
        depth: str = "standard",
        custom_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """
        Analizza un'immagine con il miglior backend disponibile.

        Args:
            image_path: percorso file immagine
            depth: "quick", "standard", "deep"
            custom_prompt: prompt personalizzato (sovrascrive il default)
        Returns:
            (testo analisi, metadata backend)
        """
        path = Path(image_path)
        if not path.exists():
            return "Immagine non trovata.", {"error": "file_not_found"}

        prompt = custom_prompt or self.ANALYSIS_PROMPTS.get(depth, self.ANALYSIS_PROMPTS["standard"])

        metadata = {
            "image_path": str(path),
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "depth": depth,
        }

        # Prova Gemini Vision
        if settings.GOOGLE_API_KEY:
            text, ok = self._analyze_gemini(path, prompt)
            if ok:
                metadata["backend"] = "gemini_vision"
                self.last_metadata = metadata
                return text, metadata

        # Prova DeepSeek VL2
        if settings.DEEPSEEK_API_KEY:
            text, ok = self._analyze_deepseek(path, prompt)
            if ok:
                metadata["backend"] = "deepseek_vl2"
                self.last_metadata = metadata
                return text, metadata

        # Fallback: OCR puro
        if settings.OCR_ENABLED:
            text = self._ocr_fallback(path)
            if text:
                metadata["backend"] = "ocr_only"
                self.last_metadata = metadata
                return f"[Analisi OCR — solo testo estratto]\n\n{text}", metadata

        return "Nessun backend vision disponibile. Configura GOOGLE_API_KEY o DEEPSEEK_API_KEY.", metadata

    def _analyze_gemini(self, path: Path, prompt: str) -> Tuple[str, bool]:
        """Analisi con Google Gemini Vision."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.GOOGLE_API_KEY)

            # Carica immagine
            uploaded = genai.upload_file(str(path))
            model = genai.GenerativeModel("gemini-2.0-flash-lite")

            start = time.time()
            response = model.generate_content([prompt, uploaded])
            latency = time.time() - start

            text = response.text or ""
            logger.info(f"Gemini Vision: analisi completata in {latency:.1f}s")
            return text, bool(text.strip())

        except Exception as e:
            logger.warning(f"Gemini Vision fallito: {e}")
            return "", False

    def _analyze_deepseek(self, path: Path, prompt: str) -> Tuple[str, bool]:
        """Analisi con DeepSeek-VL2."""
        try:
            import requests

            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = path.suffix.lower()
            media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
            media_type = media_types.get(ext, "image/jpeg")

            headers = {
                "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "deepseek-vl2",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": 2000,
                "temperature": 0.7,
            }

            start = time.time()
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json=payload, headers=headers, timeout=60,
            )
            latency = time.time() - start

            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"]
                logger.info(f"DeepSeek-VL2: analisi completata in {latency:.1f}s")
                return text, bool(text.strip())

            return "", False

        except Exception as e:
            logger.warning(f"DeepSeek-VL2 fallito: {e}")
            return "", False

    def _ocr_fallback(self, path: Path) -> str:
        """Fallback: estrazione testo con OCR."""
        try:
            from PIL import Image
            import pytesseract

            img = Image.open(str(path))
            text = pytesseract.image_to_string(img, lang="ita+eng")
            return text.strip()
        except Exception as e:
            logger.debug(f"OCR fallback fallito: {e}")
            return ""
