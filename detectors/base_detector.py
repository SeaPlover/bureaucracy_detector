'''
Единый интерфейс — чтобы main.py мог работать со всеми
тремя одинаково, не зная деталей реализации каждого.
'''

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import nltk

class BaseDetector(ABC):
    """Абстрактный класс детектора канцеляризмов"""

    def _split_into_sentences(self, text: str) -> List[str]:
        """Вспомогательный метод разбиения текста на предложения"""
        try:
            return nltk.sent_tokenize(text, language="russian")
        except:
            # Фолбэк на случай проблем с токенизатором nltk
            return [s.strip() for s in text.split('.') if s.strip()]

    @abstractmethod
    def detect(self, text: str, preprocessed: Optional[List] = None) -> List[Dict]:
        pass