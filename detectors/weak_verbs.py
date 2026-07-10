import os
import re
import json
from typing import List, Dict, Optional, Set
from .base_detector import BaseDetector
from llm.client import LLMClient
from core.white_list import WhiteList
from core.postprocessor import LLMResponsePostprocessor


class WeakVerbsDetector(BaseDetector):
    """Детектор слабых глаголов с пакетной обработкой"""

    MODAL_PREFIX_LEMMAS = {'должен', 'обязан', 'готов', 'рад', 'достоин'}
    STATE_WORDS = {'надо', 'необходимо', 'можно', 'приятно', 'нельзя', 'возможно'}
    PREPOSITIONS = {'в', 'на', 'с', 'к', 'у', 'о', 'об', 'от', 'до', 'из', 'за',
                    'через', 'по', 'под', 'над', 'перед', 'при', 'без', 'для'}

    def __init__(self, llm_client: LLMClient, prompt_template: str, white_list: WhiteList, weak_verbs_txt: str = None):
        self.llm = llm_client
        self.prompt_template = prompt_template
        self.white_list = white_list
        self.weak_verbs = self._load_weak_verbs(weak_verbs_txt)
        if not self.weak_verbs:
            self.weak_verbs = {'быть', 'стать', 'иметь', 'становление', 'являться', 'осуществлять', 'производить'}

    def _load_weak_verbs(self, txt_path: str = None) -> Set[str]:
        verbs = set()
        if txt_path and os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        v = line.strip().lower()
                        if v and not v.startswith('#'):
                            verbs.add(v)
            except:
                pass
        return verbs

    def detect(self, text: str, preprocessed: Optional[List] = None) -> List[Dict]:
        if preprocessed is None:
            from core.preprocessor import TextPreprocessor
            preprocessed = TextPreprocessor().process(text)

        candidates = []

        for sentence_tokens in preprocessed:
            sentence_text = " ".join([t["token"] for t in sentence_tokens])

            for i, token_data in enumerate(sentence_tokens):
                word = token_data["token"]
                lemma = (token_data.get("lemma") or word.lower()).lower()

                if (lemma in self.weak_verbs or lemma in self.STATE_WORDS) and not self.white_list.is_excluded(word,
                                                                                                               lemma):
                    construction = self._extract_construction(sentence_tokens, i)
                    if construction:
                        candidates.append({
                            "line": sentence_text,
                            "words": construction
                        })

        if not candidates:
            return []

        # Батчинг запроса к LLM
        batch_tasks = [{"id": idx, "phrase": c["words"], "context": c["line"]} for idx, c in enumerate(candidates)]

        prompt = (
            f"ИНСТРУКЦИЯ И КРИТЕРИИ:\n{self.prompt_template}\n\n"
            f"ЗАДАНИЕ:\nПодбери сильные замены для конструкций:\n{json.dumps(batch_tasks, ensure_ascii=False)}\n\n"
            f"Формат ответа: строго JSON-массив объектов [{{\"id\": 0, \"suggestions\": [\"замена1\", \"замена2\"]}}]. Без markdown."
        )

        findings = []
        try:
            llm_response = self.llm.generate(prompt)
            clean_str = LLMResponsePostprocessor.extract_json_string(llm_response)
            results_dict = {item["id"]: item.get("suggestions", []) for item in json.loads(clean_str) if "id" in item}
        except Exception as e:
            print(f"[Ошибка пакетного LLM в WeakVerbs]: {e}")
            results_dict = {}

        for idx, c in enumerate(candidates):
            sug = results_dict.get(idx, ["изменить на активный глагол"])
            findings.append({
                "line": c["line"],
                "words": c["words"],
                "suggestions": sug[:2]
            })

        return findings[:30]

    def _extract_construction(self, tokens: List[Dict], start_idx: int) -> Optional[str]:
        # Модальный префикс слева
        left_words = []
        if start_idx > 0:
            prev = tokens[start_idx - 1]
            prev_lemma = (prev.get('lemma') or prev['token']).lower()
            if prev_lemma in self.MODAL_PREFIX_LEMMAS or prev['token'].lower() in self.MODAL_PREFIX_LEMMAS:
                left_words.append(prev['token'])

        construction_tokens = left_words + [tokens[start_idx]['token']]
        found_target = False

        for j in range(start_idx + 1, len(tokens)):
            next_tok = tokens[j]
            next_word = next_tok['token']
            next_pos = next_tok.get('pos', '')

            if next_word.lower() in self.PREPOSITIONS:
                if len(construction_tokens) < 5:
                    construction_tokens.append(next_word)
                    continue
                else:
                    break

            if next_pos in ('NOUN', 'NOUN_PROPN', 'INFN'):
                construction_tokens.append(next_word)
                found_target = True
                break

            if next_pos in ('ADJ', 'ADJF', 'ADJS'):
                construction_tokens.append(next_word)
                found_target = True
                if j + 1 < len(tokens) and tokens[j + 1]['pos'] == 'NOUN':
                    construction_tokens.append(tokens[j + 1]['token'])
                break

            if len(construction_tokens) < 5:
                construction_tokens.append(next_word)
            else:
                break

        if found_target and len(construction_tokens) > 1:
            return " ".join(construction_tokens)
        return None