"""
Модуль evaluation/metrics.py для измерения метрик
"""

import re
from typing import List, Dict, Set
import pymorphy3

morph = pymorphy3.MorphAnalyzer()


def get_lemmas_set(text: str) -> Set[str]:
    """Извлекает множество лемм из текста"""
    if not text:
        return set()
    words = re.findall(r'[\w-]+', text.lower())
    lemmas = set()
    for w in words:
        try:
            lemmas.add(morph.parse(w)[0].normal_form)
        except:
            lemmas.add(w)
    return {l for l in lemmas if len(l) > 1}


def normalize_phrase(phrase: str) -> str:
    """
    Нормализует фразу для сравнения:
    - убирает предлоги
    - убирает прилагательные
    - оставляет только ключевые существительные
    """
    if not phrase:
        return ""

    # Убираем предлоги
    prepositions = {'в', 'на', 'с', 'к', 'у', 'о', 'об', 'от', 'до', 'из', 'за',
                    'через', 'по', 'под', 'над', 'перед', 'при', 'без', 'для',
                    'ради', 'вокруг', 'около', 'и', 'а', 'но', 'или', 'либо'}

    words = phrase.lower().split()
    # Оставляем только слова длиннее 3 букв и не предлоги
    key_words = [w for w in words if len(w) > 3 and w not in prepositions]

    # Берем первые 3 ключевых слова
    return " ".join(key_words[:3])


def calculate_accuracy_per_sentence(expected: List[Dict], predicted: List[Dict]) -> Dict:
    """
    Сопоставляет эталоны и ответы ИИ с улучшенным сравнением.
    """
    type_mapping = {
        'цепочка отглагольных существительных': 'verbal_noun',
        'отглагольное существительное': 'verbal_noun',
        'отглагольные сущ.': 'verbal_noun',
        'отглагольные': 'verbal_noun',
        'verbal_noun': 'verbal_noun',
        'слабый глагол': 'weak_verb',
        'слабые глаголы': 'weak_verb',
        'weak_verb': 'weak_verb',
        'заимствование': 'borrowing',
        'заимствования': 'borrowing',
        'borrowing': 'borrowing'
    }

    # Готовим эталоны
    exp_list = []
    for item in expected:
        phrase = item.get('words') or item.get('word') or ''
        raw_type = item.get('type', 'unknown')
        c_type = type_mapping.get(str(raw_type).lower().strip(), 'unknown')

        # Для отглагольных цепочек используем нормализованную фразу
        if c_type == 'verbal_noun':
            norm_phrase = normalize_phrase(phrase)
            lemmas = get_lemmas_set(norm_phrase) if norm_phrase else get_lemmas_set(phrase)
        else:
            lemmas = get_lemmas_set(phrase)

        if lemmas:
            exp_list.append({
                'lemmas': lemmas,
                'type': c_type,
                'found': False,
                'original': phrase
            })

    # Готовим ответы ИИ
    pred_list = []
    for item in predicted:
        phrase = item.get('words') or item.get('word') or ''
        raw_type = item.get('type', 'unknown')
        p_type = type_mapping.get(str(raw_type).lower().strip(), 'unknown')

        # Для отглагольных цепочек используем нормализованную фразу
        if p_type == 'verbal_noun':
            norm_phrase = normalize_phrase(phrase)
            lemmas = get_lemmas_set(norm_phrase) if norm_phrase else get_lemmas_set(phrase)
        else:
            lemmas = get_lemmas_set(phrase)

        if lemmas:
            pred_list.append({
                'lemmas': lemmas,
                'type': p_type,
                'matched': False,
                'original': phrase
            })

    correct = 0
    correct_by_type = {}
    expected_by_type = {}
    unmatched_snippets = []

    # Считаем ожидания по типам
    for e_item in exp_list:
        t = e_item['type']
        expected_by_type[t] = expected_by_type.get(t, 0) + 1

    # Сопоставляем
    for p_item in pred_list:
        matched = False
        for e_item in exp_list:
            if p_item['type'] != e_item['type']:
                continue
            intersection = p_item['lemmas'] & e_item['lemmas']
            if intersection:
                matched = True
                if not e_item['found']:
                    e_item['found'] = True
                    p_item['matched'] = True
                    correct += 1
                    c_type = e_item['type']
                    correct_by_type[c_type] = correct_by_type.get(c_type, 0) + 1
                else:
                    p_item['matched'] = True
                break

        if not matched:
            unmatched_snippets.append(p_item['original'])

    # Подсчет ложных срабатываний
    fps = sum(1 for p in pred_list if not p['matched'])

    return {
        'total_expected': len(exp_list),
        'found_correctly': correct,
        'false_positives': fps,
        'expected_by_type': expected_by_type,
        'correct_by_type': correct_by_type,
        'unmatched': unmatched_snippets
    }