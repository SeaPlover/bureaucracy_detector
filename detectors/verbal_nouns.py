import re
import json
from typing import List, Dict, Optional
from .base_detector import BaseDetector
from llm.client import LLMClient
from core.white_list import WhiteList
from core.postprocessor import LLMResponsePostprocessor


class VerbalNounsDetector(BaseDetector):
    """Детектор отглагольных существительных с пакетной обработкой через LLM"""

    # Расширенные суффиксы
    VERBAL_SUFFIXES = {
        'тель', 'итель', 'ник', 'енник', 'льник', 'щик', 'чик', 'льщик',
        'ок', 'к', 'ушк', 'шк', 'чк', 'ние', 'ение', 'ание', 'ование', 'нье', 'анье', 'енье',
        'тие', 'тье', 'ция', 'ация', 'яция', 'изация', 'фикация', 'утие', 'ствие', 'звание',
        'вание', 'оение', 'ча', 'ба', 'жа', 'льба',
    }

    # Бессуффиксальные префиксы
    VERBAL_PREFIXES = ['за', 'от', 'вы', 'до', 'при', 'пере', 'на', 'с', 'под']

    ALLOWED_SEPARATORS = {'в', 'на', 'с', 'к', 'у', 'о', 'об', 'от', 'до',
                          'из', 'за', 'через', 'по', 'под', 'над', 'перед',
                          'при', 'без', 'для', 'ради', 'вокруг', 'около',
                          'и', 'а', 'но', 'или', 'либо', 'же'}

    def __init__(self, llm_client: LLMClient, prompt_template: str, white_list: WhiteList):
        self.llm = llm_client
        self.prompt_template = prompt_template
        self.white_list = white_list
        # Импортируем pymorphy3 для анализа
        import pymorphy3
        self.morph = pymorphy3.MorphAnalyzer()

    def detect(self, text: str, preprocessed: Optional[List] = None) -> List[Dict]:
        if preprocessed is None:
            from core.preprocessor import TextPreprocessor
            preprocessed = TextPreprocessor().process(text)

        candidates = []

        # 1. Сбор всех локальных кандидатов по тексту
        for sentence_tokens in preprocessed:
            sentence_text = " ".join([t["token"] for t in sentence_tokens])
            chains = self._find_noun_chains(sentence_tokens)

            for chain in chains:
                phrase = chain['phrase']
                # Проверяем белый список для всей фразы
                if not self._is_phrase_excluded(phrase, chain['lemmas']):
                    candidates.append({
                        "line": sentence_text,
                        "words": phrase,
                        "lemmas": chain['lemmas'],
                        "noun_count": chain.get('noun_count', 0)
                    })

        if not candidates:
            return []

        # 2. Формируем ОДИН пакетный промпт для LLM
        batch_tasks = [{"id": idx, "phrase": c["words"], "context": c["line"]} for idx, c in enumerate(candidates)]

        prompt = (
            f"ИНСТРУКЦИЯ И КРИТЕРИИ:\n{self.prompt_template}\n\n"
            f"ЗАДАНИЕ:\nПроанализируй список кандидатов ниже и подбери для каждого 1-2 варианта замены.\n"
            f"Входные данные: {json.dumps(batch_tasks, ensure_ascii=False)}\n\n"
            f"Выдай ответ СТРОГО в формате JSON-массива объектов вида:\n"
            f"[{{\"id\": 0, \"suggestions\": [\"вариант1\", \"вариант2\"]}}]\n"
            f"Не добавляй markdown код-блоки (```json) и посторонний текст."
        )

        findings = []
        try:
            llm_response = self.llm.generate(prompt)
            clean_str = LLMResponsePostprocessor.extract_json_string(llm_response)
            results_dict = {item["id"]: item.get("suggestions", []) for item in json.loads(clean_str) if "id" in item}
        except Exception as e:
            print(f"[Ошибка пакетного LLM в VerbalNouns]: {e}")
            results_dict = {}

        # 3. Сборка финального результата
        for idx, c in enumerate(candidates):
            suggestions = results_dict.get(idx, [self._heuristic_suggestion(c["lemmas"])])
            findings.append({
                "line": c["line"],
                "words": c["words"],
                "suggestions": suggestions[:2]
            })

        return findings[:30]

    def _is_verbal_noun(self, lemma: str) -> bool:
        """Проверяет, является ли слово отглагольным существительным"""
        word = lemma.lower()

        # Проверка по суффиксам
        for suffix in self.VERBAL_SUFFIXES:
            if word.endswith(suffix):
                return True

        # Бессуффиксальные образования
        for prefix in self.VERBAL_PREFIXES:
            if word.startswith(prefix) and len(word) >= 4:
                stem = word[len(prefix):]
                if len(stem) >= 2:
                    return True

        # Дополнительная проверка: если слово оканчивается на "а" и является существительным
        try:
            parse = self.morph.parse(word)[0]
            if parse.tag.POS == 'NOUN':
                # Проверяем, есть ли однокоренной глагол
                # Упрощенная проверка: если слово оканчивается на типичные суффиксы
                if word.endswith(('а', 'я')) and len(word) > 3:
                    return True
        except:
            pass

        return False

    def _is_phrase_excluded(self, phrase: str, lemmas: List[str]) -> bool:
        """Проверяет, исключена ли фраза по белого списку"""
        # Проверяем каждое слово во фразе
        for lemma in lemmas:
            if self.white_list.is_excluded('', lemma):
                return True
        return False

    def _find_noun_chains(self, tokens: List[Dict]) -> List[Dict]:
        """Находит цепочки отглагольных существительных"""
        chains = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            lemma = (token.get('lemma') or token['token']).lower()
            pos = token.get('pos', '')

            # Если это не существительное или не отглагольное — пропускаем
            if pos != 'NOUN' or not self._is_verbal_noun(lemma):
                i += 1
                continue

            current_chain = [token['token']]
            current_lemmas = [lemma]
            current_pos = [pos]
            j = i + 1

            # Собираем цепочку
            while j < len(tokens):
                next_token = tokens[j]
                next_lemma = (next_token.get('lemma') or next_token['token']).lower()
                next_pos = next_token.get('pos', '')
                next_word = next_token['token'].lower()

                # Если это существительное и отглагольное — добавляем
                if next_pos == 'NOUN' and self._is_verbal_noun(next_lemma):
                    current_chain.append(next_token['token'])
                    current_lemmas.append(next_lemma)
                    current_pos.append(next_pos)
                    j += 1
                    continue

                # Если это прилагательное — добавляем (оно часть цепочки)
                if next_pos in ('ADJ', 'ADJF', 'ADJS'):
                    current_chain.append(next_token['token'])
                    current_lemmas.append(next_lemma)
                    current_pos.append(next_pos)
                    j += 1
                    continue

                # Если это предлог или союз — добавляем
                if next_word in self.ALLOWED_SEPARATORS:
                    current_chain.append(next_token['token'])
                    current_lemmas.append(next_lemma)
                    current_pos.append(next_pos)
                    j += 1
                    continue

                # Если это предлог с союзом — добавляем
                if next_word in ['и', 'а', 'но', 'или', 'либо']:
                    current_chain.append(next_token['token'])
                    current_lemmas.append(next_lemma)
                    current_pos.append(next_pos)
                    j += 1
                    continue

                # Иначе — разрываем цепочку
                break

            # Проверяем, сколько в цепочке существительных
            noun_count = sum(1 for p in current_pos if p == 'NOUN')

            # Если в цепочке 2+ существительных — сохраняем
            if noun_count >= 2:
                chains.append({
                    'phrase': " ".join(current_chain),
                    'lemmas': current_lemmas,
                    'pos': current_pos,
                    'noun_count': noun_count
                })

            i = j

        return chains

    def _heuristic_suggestion(self, lemmas: List[str]) -> str:
        """Эвристическое предложение замены (без LLM)"""
        if not lemmas:
            return "переформулировать фразу"

        # Пробуем образовать глагол от первого существительного
        first_lemma = lemmas[0]

        for suffix in self.VERBAL_SUFFIXES:
            if first_lemma.endswith(suffix):
                stem = first_lemma[:-len(suffix)]
                if stem and len(stem) > 2:
                    if stem.endswith('к'):
                        return f"{stem[:-1]}ать"
                    elif stem.endswith('н'):
                        return f"{stem}уть"
                    else:
                        return f"{stem}ть"

        return f"использовать глагол вместо '{first_lemma}'"