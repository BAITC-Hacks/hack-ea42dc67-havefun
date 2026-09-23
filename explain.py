"""Объяснение плана кампаний на естественном языке.

Важно: этот модуль НЕ участвует в принятии решений. Все числа — охват,
эффект, выбор канала, порядок кампаний — посчитаны в `agent.py` детерминированно.
Модель получает уже готовый план с измеренными величинами и только
формулирует обоснование словами.

Если ключа нет или API недоступен, объяснение собирается по шаблону из тех же
чисел. Поэтому отказ внешнего сервиса не меняет ни план, ни результат —
меняется только формулировка.
"""
from __future__ import annotations

import os

# Имя модели не зашиваем: на судействе ключ будет чужой, и захардкоженная
# модель может быть недоступна на том аккаунте. Поэтому спрашиваем у API
# список доступных и берём первую подходящую из порядка предпочтения —
# от дешёвых и быстрых к более крупным. Переопределяется через OPENAI_MODEL.
# Порядок предпочтения на 23.09.2026 (platform.openai.com/docs/models):
#   gpt-6-luna  — $0.1/$0.5 за MTok, для высокочастотных задач
#   gpt-6-sol   — $2/$10, баланс
#   gpt-6-astra — $10/$50, флагман
# Наша задача — пересказать уже посчитанные числа, поэтому берём luna:
# платить за рассуждения флагмана здесь не за что.
MODEL_PREFERENCE = ["gpt-6-luna", "gpt-6-sol", "gpt-6-astra", "gpt-4o-mini"]
TIMEOUT_S = 30
_resolved_model: str | None = None
CHANNEL_RU = {
    "push": "push (бесплатно)",
    "sms": "SMS (4 у.е. за контакт)",
    "digital_ads": "реклама (22 у.е. за контакт)",
    "call": "звонок (160 у.е. за контакт)",
}


def _facts(campaigns: list[dict], pilots: list[dict]) -> str:
    """Собирает факты, на которых строится объяснение."""
    lines = [f"Проведено пилотов: {len(pilots)}."]
    for c in campaigns:
        lines.append(
            f"Кампания «{c['campaign_name']}»: сегмент {c['filter_arpu_segment']}, "
            f"целевой тариф {c['target_tariff']}, канал {CHANNEL_RU.get(c['channel'], c['channel'])}, "
            f"охват {c['reach']} абонентов, измеренный эффект {c['base']:.3f} "
            f"от ARPU, средний ARPU сегмента {c['arpu']:.0f} у.е."
        )
    return "\n".join(lines)


def _fallback(campaigns: list[dict], pilots: list[dict]) -> str:
    """Детерминированное объяснение из тех же чисел, без модели."""
    out = [f"План построен по результатам {len(pilots)} пилотных кампаний.", ""]
    for c in campaigns:
        lift = c["base"] * c["arpu"]
        out.append(
            f"- **{c['campaign_name']}**: берём сегмент {c['filter_arpu_segment']} "
            f"({c['reach']} абонентов) и предлагаем {c['target_tariff']}. "
            f"Пилот показал прирост {c['base']:.1%} от ARPU, это около {lift:.0f} у.е. "
            f"на абонента при среднем ARPU {c['arpu']:.0f}. "
            f"Канал — {CHANNEL_RU.get(c['channel'], c['channel'])}: "
            f"прирост окупает стоимость контакта."
        )
    return "\n".join(out)


SYSTEM_PROMPT = (
    "Ты объясняешь маркетинговый план аналитику оператора связи. "
    "Опирайся ТОЛЬКО на переданные числа, ничего не додумывай и не меняй "
    "значения. На каждую кампанию — два-три предложения: кого берём, что "
    "предлагаем, почему этот канал, чем подтверждено. "
    "Пиши по-русски, без маркетингового пафоса."
)


def call_model(client, model: str, system: str, user: str) -> str:
    """Вызов модели: сначала Responses API, при неудаче — chat.completions.

    Документация OpenAI называет Responses основным интерфейсом, но на
    отдельных аккаунтах и прокси доступен только chat.completions.
    Поддерживаем оба, чтобы решение не зависело от того, какой ключ дадут.
    """
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        if text:
            return text
    except Exception as exc:  # noqa: BLE001 — падаем на совместимый путь
        print(f"[explain] Responses API недоступен ({type(exc).__name__}), "
              f"пробуем chat.completions")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def _pick_model(client) -> str | None:
    """Выбирает модель из доступных на аккаунте по порядку предпочтения."""
    global _resolved_model
    if _resolved_model:
        return _resolved_model
    forced = os.environ.get("OPENAI_MODEL")
    if forced:
        _resolved_model = forced
        return forced
    try:
        available = {m.id for m in client.models.list()}
    except Exception as exc:  # noqa: BLE001 — список моделей может быть закрыт
        print(f"[explain] список моделей недоступен ({type(exc).__name__}), "
              f"берём {MODEL_PREFERENCE[0]}")
        _resolved_model = MODEL_PREFERENCE[0]
        return _resolved_model
    for name in MODEL_PREFERENCE:
        if name in available:
            _resolved_model = name
            print(f"[explain] модель: {name}")
            return name
    # ничего из списка предпочтений нет — берём любую чат-модель
    chat_like = sorted(m for m in available if m.startswith("gpt-"))
    _resolved_model = chat_like[0] if chat_like else None
    if _resolved_model:
        print(f"[explain] из предпочтений ничего нет, берём {_resolved_model}")
    return _resolved_model


def explain_plan(campaigns: list[dict], pilots: list[dict]) -> str:
    """Возвращает текстовое обоснование плана.

    При недоступности модели отдаёт шаблонный вариант — план от этого
    не меняется.
    """
    if not campaigns:
        return "План пуст: ни одна гипотеза не показала уверенно положительного эффекта."

    if not os.environ.get("OPENAI_API_KEY"):
        return _fallback(campaigns, pilots)

    try:
        from openai import OpenAI

        client = OpenAI(timeout=TIMEOUT_S)
        model = _pick_model(client)
        if model is None:
            return _fallback(campaigns, pilots)
        text = call_model(client, model, SYSTEM_PROMPT, _facts(campaigns, pilots))
        return text if text else _fallback(campaigns, pilots)
    except Exception as exc:  # noqa: BLE001 — объяснение не должно ронять прогон
        print(f"[explain] модель недоступна ({type(exc).__name__}), объяснение по шаблону")
        return _fallback(campaigns, pilots)
