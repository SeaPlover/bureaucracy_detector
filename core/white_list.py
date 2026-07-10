"""
Менеджер белого списка слов, которые НЕ нужно заменять
Загружается из data/white_list.txt
"""

import os
from typing import Set


class WhiteList:
    """Белый список слов-исключений"""

    def __init__(self, file_path: str = None):
        self.words = set()

        if file_path and os.path.exists(file_path):
            self._load_from_file(file_path)
        else:
            # Базовый набор, если файл не найден
            self.words = {
                'компьютер', 'бюстгальтер', 'интеллект', 'менеджмент',
                'маркетинг', 'бренд', 'интернет', 'сайт', 'алгоритм',
                'сервер', 'клиент', 'процессор', 'функция', 'реакция'
            }

    def _load_from_file(self, file_path: str):
        """Загружает белый список из файла"""
        self.words = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    self.words.add(word.lower())

    def is_excluded(self, word: str, lemma: str = None) -> bool:
        """
        Проверяет, нужно ли исключить слово из обработки

        Args:
            word: Исходное слово
            lemma: Лемма слова (опционально)

        Returns:
            True если слово должно быть исключено
        """
        word_lower = word.lower()
        if word_lower in self.words:
            return True

        if lemma and lemma.lower() in self.words:
            return True

        return False

    def add(self, word: str):
        """Добавляет слово в белый список"""
        self.words.add(word.lower())

    def remove(self, word: str):
        """Удаляет слово из белого списка"""
        self.words.discard(word.lower())