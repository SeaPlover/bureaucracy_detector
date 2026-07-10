import os
import re
import json
from typing import List, Dict, Optional, Set
from .base_detector import BaseDetector
from llm.client import LLMClient
from core.white_list import WhiteList
from core.postprocessor import LLMResponsePostprocessor


class BorrowingsDetector(BaseDetector):
    """Детектор заимствований со сквозным пакетным запросом"""

    def __init__(self, llm_client: LLMClient, prompt_template: str, white_list: WhiteList,
                 borrow_list_path: str = None):
        self.llm = llm_client
        self.prompt_template = prompt_template
        self.white_list = white_list
        self.borrow_words = self._load_borrow_list(borrow_list_path) if borrow_list_path else set()

    def _load_borrow_list(self, file_path: str) -> Set[str]:
        borrow_set = set()
        if not os.path.exists(file_path): return borrow_set
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip().lower()
                    if w and not w.startswith('#'): borrow_set.add(w)
        except:
            pass
        return borrow_set

    def detect(self, text: str, preprocessed: Optional[List] = None) -> List[Dict]:
        if preprocessed is None:
            from core.preprocessor import TextPreprocessor
            preprocessed = TextPreprocessor().process(text)

        all_candidates = []

        # Собираем всех кандидатов по всем предложениям
        for sentence_tokens in preprocessed:
            sentence_text = " ".join([t["token"] for t in sentence_tokens])

            for token_data in sentence_tokens:
                word = token_data["token"]
                lemma = (token_data.get("lemma") or word.lower()).lower()

                if not re.match(r'^[а-яА-Яa-zA-Z\-]+$', word): continue

                if lemma in self.borrow_words and not self.white_list.is_excluded(word, lemma):
                    all_candidates.append({
                        "context": sentence_text,
                        "word": word,
                        "lemma": lemma
                    })

        if not all_candidates:
            return []

        # Формируем единый батч-запрос
        batch_tasks = [{"id": idx, "phrase": c["word"], "context": c["context"]} for idx, c in
                       enumerate(all_candidates)]

        prompt = (
            f"ИНСТРУКЦИЯ И КРИТЕРИИ:\n{self.prompt_template}\n\n"
            f"ЗАДАНИЕ:\nПодбери русские аналоги для заимствований:\n{json.dumps(batch_tasks, ensure_ascii=False)}\n\n"
            f"Ответ выдай СТРОГО в формате JSON-массива: [{{\"id\": 0, \"suggestions\": [\"аналог1\", \"аналог2\"]}}]. Без markdown."
        )

        findings = []
        try:
            llm_response = self.llm.generate(prompt)
            clean_str = LLMResponsePostprocessor.extract_json_string(llm_response)
            results_dict = {item["id"]: item.get("suggestions", []) for item in json.loads(clean_str) if "id" in item}
        except Exception as e:
            print(f"[Ошибка пакетного LLM в Borrowings]: {e}")
            results_dict = {}

        for idx, c in enumerate(all_candidates):
            sug = results_dict.get(idx, [f"заменить слово '{c['word']}' русским аналогом"])
            findings.append({
                "line": c["context"],
                "words": c["word"],
                "suggestions": sug[:2]
            })

        return findings[:30]