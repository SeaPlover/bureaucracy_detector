"""
Главный файл для запуска пайплайна анализа текста.
Использует детекторы из папки detectors/
"""

import sys
import yaml
import json
from typing import List, Dict
import os
import re

from core.preprocessor import TextPreprocessor
from core.white_list import WhiteList
from detectors.verbal_nouns import VerbalNounsDetector
from detectors.weak_verbs import WeakVerbsDetector
from detectors.borrowings import BorrowingsDetector
from llm.client import LLMClient


def load_config(config_path: str = "config.yaml") -> Dict:
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_prompt(prompt_path: str) -> str:
    """Загружает промпт из файла"""
    if not prompt_path or not os.path.exists(prompt_path):
        return ""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def clean_for_matching(text: str) -> str:
    """Очищает строку от знаков препинания и пробелов для надежного сопоставления"""
    return re.sub(r'[^\w]', '', text.lower())


def main(text: str, config_path: str = "config.yaml") -> Dict:
    """Основная функция обработки текста"""
    config = load_config(config_path)

    # Инициализируем общие компоненты
    llm_client = LLMClient(config["llm"])
    white_list = WhiteList(config.get("white_list_path"))

    # Предобработка текста (Разбивка на предложения и токены через Natasha)
    preprocessor = TextPreprocessor()
    preprocessed_data = preprocessor.process(text)

    # Загружаем базовые инструкции из файлов промптов
    prompts = config.get("prompts", {})
    vn_prompt = load_prompt(prompts.get("verbal_nouns"))
    wv_prompt = load_prompt(prompts.get("weak_verbs"))
    br_prompt = load_prompt(prompts.get("borrowings"))

    # Инициализируем детекторы
    detectors = {
        "цепочка отглагольных существительных": VerbalNounsDetector(llm_client, vn_prompt, white_list),
        "слабый глагол": WeakVerbsDetector(llm_client, wv_prompt, white_list, config.get("weak_verbs_path")),
        "заимствование": BorrowingsDetector(llm_client, br_prompt, white_list, config.get("borrow_list_path"))
    }

    all_results = []
    source_lines = text.split('\n')

    # Сопоставляем текст предложения (как его реконструируют детекторы —
    # " ".join(токенов)) с его порядковым номером в preprocessed_data.
    # Это надёжнее text.find(): не зависит от точного совпадения символов.
    sentence_index_by_text = {
        " ".join([t["token"] for t in sent_tokens]): idx
        for idx, sent_tokens in enumerate(preprocessed_data, start=1)
    }

    for det_type, detector in detectors.items():
        findings = detector.detect(text, preprocessed=preprocessed_data)
        for f in findings:
            words = f.get("words", "")
            sentence_text = f.get("line", "")  # детекторы кладут сюда текст предложения
            sent_num = sentence_index_by_text.get(sentence_text, 0)

            all_results.append({
                 "line_num": sent_num,
                 "words": words,
                 "type": det_type,
                 "suggestions": f.get("suggestions", [])
            })

    # Сортируем по номеру предложения (без хрупкого повторного text.find())
    all_results.sort(key=lambda x: x["line_num"])

    return {
        "text": text,
        "results": all_results,
        "total_found": len(all_results)
    }


def print_results(results: Dict):
    """Красиво выводит результаты строго по формату пользователя"""
    print("\n" + "=" * 80)
    print(f"АНАЛИЗ ТЕКСТА - НАЙДЕНО {results['total_found']} КАНЦЕЛЯРИЗМОВ")
    print("=" * 80)

    for finding in results['results']:
        suggestions_str = " / ".join(finding['suggestions']) if finding['suggestions'] else "переформулировать фразу"

        print(f"\nПредложение № {finding['line_num']}")
        print(f"Канцеляризм: {finding['words']}")
        print(f"Вид: {finding['type']}")
        print(f"Предложение по замене: {suggestions_str}")


if __name__ == "__main__":
    print("=== ИНТЕРАКТИВНЫЙ ДЕТЕКТОР КАНЦЕЛЯРИЗМОВ ===")
    print("Введите или вставьте текст для анализа (для завершения ввода нажмите Enter на пустой строке):")

    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break

    input_text = "\n".join(lines)

    if input_text.strip():
        print("\n[Процесс] Анализируем текст...")
        res = main(input_text)
        print_results(res)
    else:
        print(" Ошибка: Пустой ввод.")