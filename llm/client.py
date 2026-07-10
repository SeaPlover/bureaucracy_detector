"""
Клиент для работы с LLM через официальный Hugging Face Inference Router
(OpenAI-совместимый эндпоинт router.huggingface.co)
"""
import os
from typing import Dict
from openai import OpenAI


class LLMClient:
    """Обёртка для вызовов моделей через Hugging Face Router"""

    HF_ROUTER_URL = "https://router.huggingface.co/v1"

    def __init__(self, config: Dict):
        self.config = config
        self.model = config.get("model", "meta-llama/Llama-3.1-8B-Instruct")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 3000)

        # Ключ ищем в конфиге, а если не задан — в переменной окружения HF_TOKEN
        # (так рекомендует сама документация HF, и так токен не светится в yaml)
        api_key = config.get("api_key") or os.environ.get("HF_TOKEN")
        if not api_key or api_key == "...":
            print(
                "[WARNING] HF_TOKEN не найден! Задайте export HF_TOKEN=hf_... "
                "или заполните api_key в config.yaml (не оставляйте '...')."
            )

        self.client = OpenAI(
            base_url=self.HF_ROUTER_URL,   # всегда HF, никакого OpenRouter
            api_key=api_key,
            timeout=60.0,
        )

    def generate(self, prompt: str) -> str:
        import time
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,   # тот же ключ, что и в __init__ — без "hf_model"
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Ты — лингвист, эксперт по очистке текста от канцеляризмов. "
                                "Ты ВСЕГДА отвечаешь строго в формате JSON, без рассуждений, "
                                "вступлений и markdown-оберток (без ```json)."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(self.temperature),
                    max_tokens=int(self.max_tokens),
                )
                return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                if ("429" in error_msg or "402" in error_msg) and attempt < max_retries - 1:
                    wait_time = 15.0  # для 402/429 ждём чуть дольше, чем раньше
                    print(
                        f"\n[Лимит]: Ошибка {error_msg[:50]}... Ждём {wait_time} сек. перед попыткой {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"\n[КРИТИЧЕСКИЙ СБОЙ В CLIENT.PY]: {e}")
                    raise e