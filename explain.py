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

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-sol")
TIMEOUT_S = 30
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
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": (
                    "Ты объясняешь маркетинговый план аналитику оператора связи. "
                    "Опирайся ТОЛЬКО на переданные числа, ничего не додумывай и не "
                    "меняй значения. На каждую кампанию — два-три предложения: "
                    "кого берём, что предлагаем, почему этот канал, чем подтверждено. "
                    "Пиши по-русски, без маркетингового пафоса."
                )},
                {"role": "user", "content": _facts(campaigns, pilots)},
            ],
        )
        text = (resp.output_text or "").strip()
        return text if text else _fallback(campaigns, pilots)
    except Exception as exc:  # noqa: BLE001 — объяснение не должно ронять прогон
        print(f"[explain] модель недоступна ({type(exc).__name__}), объяснение по шаблону")
        return _fallback(campaigns, pilots)
