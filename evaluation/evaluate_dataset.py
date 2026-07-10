"""
Скрипт для запуска бенчмарка.
"""
import json
import os
import sys
import time
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main import main as run_pipeline
from metrics import calculate_accuracy_per_sentence


def load_benchmark_dataset(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл датасета не найден: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_pipeline_with_retry(text: str, max_retries: int = 5, initial_delay: float = 5.0) -> Dict:
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return run_pipeline(text)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg or "wait" in err_msg or "too many requests" in err_msg:
                if attempt == max_retries: raise e
                print(f"\n[Rate Limit] Ждем {delay} сек...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e


def run_benchmark():
    dataset_path = os.path.join(current_dir, "benchmark_dataset.json")
    try:
        dataset = load_benchmark_dataset(dataset_path)
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        return

    total_expected = 0
    total_correct = 0
    total_fps = 0

    by_type_stats = {
        'verbal_noun': {'expected': 0, 'correct': 0},
        'weak_verb': {'expected': 0, 'correct': 0},
        'borrowing': {'expected': 0, 'correct': 0}
    }

    all_unmatched = []

    print(f"=== ЗАГРУЗКА БЕНЧМАРКА ===")
    print(f"Успешно загружено {len(dataset)} текстов.\n")

    for idx, item in enumerate(dataset, 1):
        print(f"Обработка текста {idx}/{len(dataset)}...", end="", flush=True)
        try:
            pipeline_output = run_pipeline_with_retry(item["text"])
            predicted = pipeline_output.get("results", [])
            print(" Успешно")
        except Exception as e:
            print(f" Пропущено: {e}")
            predicted = []

        expected = item.get("expected_findings", [])

        # Считаем метрики строго для текущего предложения
        res = calculate_accuracy_per_sentence(expected, predicted)

        total_expected += res['total_expected']
        total_correct += res['found_correctly']
        total_fps += res['false_positives']
        all_unmatched.extend(res['unmatched'])

        # Накапливаем статистику по типам
        for t in by_type_stats.keys():
            by_type_stats[t]['expected'] += res['expected_by_type'].get(t, 0)
            by_type_stats[t]['correct'] += res['correct_by_type'].get(t, 0)

        if idx < len(dataset):
            time.sleep(5.0)

    if all_unmatched:
        print("\n[ПОДСКАЗКА МЕТРИКИ] Канцеляризмы, найденные ИИ, но отсутствующие в разметке:")
        for up in list(set(all_unmatched))[:5]:
            print(f"  - \"{up}\"")

    print("\n" + "=" * 65)
    print("         СВОДНЫЙ ОТЧЕТ ДЕТАЛЬНОЙ АНАЛИТИКИ")
    print("=" * 65)

    accuracy = total_correct / total_expected if total_expected > 0 else 0
    print(f"ОБЩАЯ ТОЧНОСТЬ СИСТЕМЫ (Accuracy): {accuracy:.2%}")
    print(f"Всего должно быть найдено: {total_expected} | Найдено верно: {total_correct}")
    print(f"Пропущено моделью (FN):    {total_expected - total_correct} | Лишние срабатывания (FP): {total_fps}")
    print("-" * 65)
    print("ДЕТАЛИЗАЦИЯ ПО ТИПАМ КАНЦЕЛЯРИЗМОВ:")
    print(f"{'Тип детектора':<22} | {'Ожидалось':<10} | {'Найдено':<8} | {'Точность':<8}")
    print("-" * 65)

    type_names = {
        'verbal_noun': 'Отглагольные сущ.',
        'weak_verb': 'Слабые глаголы',
        'borrowing': 'Заимствования'
    }

    for t, stat in by_type_stats.items():
        name = type_names[t]
        exp_c = stat['expected']
        corr_c = stat['correct']
        t_acc = corr_c / exp_c if exp_c > 0 else 0
        print(f"{name:<22} | {exp_c:<10} | {corr_c:<8} | {t_acc:.2%}")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()