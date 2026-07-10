import json
import re
from typing import List, Dict, Any


class LLMResponsePostprocessor:
    """Централизованная обработка ответов LLM"""

    @staticmethod
    def extract_json_string(text: str) -> str:
        """Находит и извлекает чистую строку JSON из любого текста ИИ"""
        if not text:
            return ""

        text = text.strip()

        # Убираем блок рассуждений reasoning-моделей (<think>...</think>)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = text.strip()

        # Убираем маркдаун-обертки
        text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        # Ищем JSON (массив или объект)
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    @staticmethod
    def parse_to_list(response: str) -> List[str]:
        """
        Извлекает список вариантов замен

        Args:
            response: Ответ от LLM

        Returns:
            List[str]: Список предложений замен
        """
        if not response:
            return ["переформулировать фразу"]

        clean_str = LLMResponsePostprocessor.extract_json_string(response)
        suggestions = []

        # Пробуем парсить JSON
        try:
            data = json.loads(clean_str)

            # Случай 1: ["замена1", "замена2"]
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Случай 1а: [{"suggestions": ["замена1", "замена2"]}]
                        sug = item.get('suggestions')
                        if isinstance(sug, list):
                            suggestions.extend(str(s) for s in sug if s)
                        elif sug:
                            suggestions.append(str(sug))
                    elif item:
                        suggestions.append(str(item))
                return suggestions if suggestions else ["переформулировать фразу"]

            # Случай 2: {"suggestions": ["замена1", "замена2"]}
            if isinstance(data, dict):
                if 'suggestions' in data:
                    sug = data['suggestions']
                    if isinstance(sug, list):
                        return [str(s) for s in sug if s] if sug else ["переформулировать фразу"]
                    return [str(sug)] if sug else ["переформулировать фразу"]

                # Если ключи — это предложения, а значения — их описания
                values = [str(v) for v in data.values() if v and str(v).strip()]
                if values:
                    return values

        except json.JSONDecodeError:
            pass

        # Если JSON не спарсился — пробуем извлечь построчно из очищенного текста
        lines = clean_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Убираем маркеры списков и номера
            line = re.sub(r'^[\d\-\*•]\s*', '', line)

            # Если строка похожа на предложение замены
            if line and len(line) > 3:
                # Пропускаем служебные строки
                if any(keyword in line.lower() for keyword in ['контекст', 'слова', 'замена']):
                    continue
                # Пропускаем строки, которые выглядят как JSON
                if line.startswith('{') or line.startswith('['):
                    continue
                suggestions.append(line)

        # Если ничего не нашли, возвращаем первую непустую строку из ответа
        if not suggestions and response.strip():
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and len(line) > 3 and not line.startswith('<'):
                    suggestions.append(line)
                    break

        return suggestions if suggestions else ["переформулировать фразу"]

    @staticmethod
    def parse_to_dict(response: str) -> Dict[str, List[str]]:
        """
        Извлекает словарь {слово: [замены]} для заимствований

        Args:
            response: Ответ от LLM

        Returns:
            Dict[str, List[str]]: Словарь с заменами для каждого слова
        """
        if not response:
            return {}

        clean_str = LLMResponsePostprocessor.extract_json_string(response)
        result = {}

        # Пробуем парсить JSON
        try:
            data = json.loads(clean_str)

            # Случай 1: {"слово1": ["замена1", "замена2"], "слово2": ["замена3"]}
            if isinstance(data, dict):
                for k, v in data.items():
                    # Пропускаем служебные ключи
                    if k.lower() in ['line', 'context', 'suggestions']:
                        continue

                    # Преобразуем значение в список строк
                    if isinstance(v, list):
                        result[k.lower()] = [str(item) for item in v if item and str(item).strip()]
                    elif isinstance(v, str):
                        result[k.lower()] = [v] if v and v.strip() else []
                    else:
                        result[k.lower()] = [str(v)]

                # Если есть отдельные объекты с полями "word" и "suggestions"
                if 'word' in data and 'suggestions' in data:
                    key = str(data['word']).lower()
                    sug = data['suggestions']
                    if isinstance(sug, list):
                        result[key] = [str(s) for s in sug if s and str(s).strip()]
                    elif isinstance(sug, str):
                        result[key] = [sug] if sug and sug.strip() else []
                    # Убираем дублирующий ключ
                    result.pop('word', None)
                    result.pop('suggestions', None)

                return result

            # Случай 2: [{"word": "слово1", "suggestions": ["замена1"]}, {"word": "слово2", ...}]
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    key = item.get('word') or item.get('words') or item.get('original')
                    if not key:
                        continue

                    sug = item.get('suggestions', [])
                    if isinstance(sug, list):
                        result[str(key).lower()] = [str(s) for s in sug if s and str(s).strip()]
                    elif isinstance(sug, str):
                        result[str(key).lower()] = [sug] if sug and sug.strip() else []
                    else:
                        result[str(key).lower()] = []

                return result

        except json.JSONDecodeError:
            pass

        # Если JSON не спарсился — пробуем извлечь построчно
        lines = clean_str.strip().split('\n')
        current_word = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Ищем слово с заменой: "слово → замена" или "слово: замена"
            match = re.match(r'^([^:→\-]+)[:→\-]\s*(.+)$', line)
            if match:
                current_word = match.group(1).strip().lower()
                suggestion = match.group(2).strip()
                if current_word and suggestion:
                    if current_word not in result:
                        result[current_word] = []
                    result[current_word].append(suggestion)
                continue

            # Если текущее слово есть и это продолжение списка замен
            if current_word and line.startswith(('-', '•', '*', '–')):
                suggestion = line.lstrip('-•*– ').strip()
                if suggestion:
                    if current_word not in result:
                        result[current_word] = []
                    result[current_word].append(suggestion)

        return result


    @classmethod
    def parse_custom_text_format(cls, raw_response: str) -> List[Dict[str, Any]]:
        """
        Парсит кастомный текстовый формат модели в структурированный список словарей.
        """
        parsed_results = []

        # Разбиваем ответ на отдельные блоки (например, по "Предложение №")
        blocks = re.split(r'(?=Предложение №\s*\d+)', raw_response)

        # Карта соответствия типов для детальной аналитики
        type_mapping = {
            'слабый глагол': 'weak_verb',
            'цепочка отглагольных существительных': 'verbal_noun',
            'отглагольное существительное': 'verbal_noun',
            'заимствование': 'borrowing'
        }

        for block in blocks:
            if not block.strip():
                continue

            # Ищем ключевые поля регулярными выражениями
            words_match = re.search(r'Канцеляризм:\s*(.+)', block)
            type_match = re.search(r'Вид:\s*(.+)', block)
            sug_match = re.search(r'Предложение по замене:\s*(.+)', block)

            if words_match:
                words = words_match.group(1).strip()

                # Определяем тип
                raw_type = type_match.group(1).strip().lower() if type_match else 'unknown'
                resolved_type = type_mapping.get(raw_type, 'unknown')

                # Разбиваем варианты замен по слэшу
                suggestions = []
                if sug_match:
                    # Разделяем по слэшу, очищаем от лишних пробелов
                    suggestions = [s.strip() for s in sug_match.group(1).split('/') if s.strip()]

                parsed_results.append({
                    'words': words,
                    'type': resolved_type,
                    'suggestions': suggestions
                })

        return parsed_results