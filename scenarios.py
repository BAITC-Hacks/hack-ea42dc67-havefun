"""Проверка на мирах с ДРУГОЙ моделью эффектов.

    python scenarios.py

Главный риск кейса сформулирован в ТЗ прямым текстом: «эффекты на судействе
другие, чем в мок-среде». Хороший результат на моке ничего не доказывает,
если агент запомнил, что LOW растёт, а HIGH падает.

Этот скрипт строит миры, где структура эффектов другая или перевёрнутая,
и прогоняет в них того же агента. Проверяется не величина результата —
она в каждом мире своя, — а поведение:

  • находит ли агент прибыльные сегменты, когда они не те, что в истории;
  • удерживается ли он от убытка, когда прибыльных сегментов нет вовсе;
  • не разоряется ли на разведке.

Это ответ на вопрос «вы подогнались под мок или нет» — измерением,
а не обещанием.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

import agent as A
from environment import make_environment
from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET, _mock_fallback
from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns

SEGMENTS = ["LOW", "MID", "HIGH"]
FILTER_COLUMNS = ["filter_arpu_segment", "filter_data_segment",
                  "filter_call_segment", "filter_current_tariff"]


def build_model(dict_tariff: pd.DataFrame, effect_by_segment: dict[str, float],
                conversion: float = 0.3, rng: np.random.Generator | None = None,
                noise: float = 0.0) -> pd.DataFrame:
    """Собирает модель эффектов с заданным приростом по сегментам."""
    tariffs = list(dict_tariff["tariff_plan_code"])
    rows = []
    for src in tariffs:
        for dst in tariffs:
            if src == dst:
                continue
            for seg in SEGMENTS:
                base = effect_by_segment.get(seg, 0.0)
                if noise and rng is not None:
                    base += float(rng.normal(0.0, noise))
                rows.append({
                    "tariff_plan_code_from": src,
                    "tariff_plan_code_to": dst,
                    "arpu_segment": seg,
                    "arpu_change_pct": base,
                    "conversion_rate": conversion,
                })
    return pd.DataFrame(rows)


def run_world(model: pd.DataFrame, seed: int) -> dict:
    """Прогоняет агента в мире с заданной моделью эффектов."""
    dict_tariff = pd.read_csv("data/dict_tariff.csv")
    profile = pd.read_csv("customer_profile.csv")

    env, internals = make_environment(
        customer_profile=profile, impact_model=model, dict_tariff=dict_tariff,
        channels=CHANNELS, total_budget=TOTAL_BUDGET,
        max_total_contacts=MAX_TOTAL_CONTACTS, fallback_predict=_mock_fallback,
        seed=seed,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            final = A.Agent().act(env)
        except Exception:      # noqa: BLE001 — падение агента тоже результат теста
            final = []
        final = sanitize_campaigns(final, env.tariffs)[:MAX_CAMPAIGNS]
        pilots = internals.executed_pilot_campaigns()

        all_campaigns = pd.DataFrame(pilots + final)
        if all_campaigns.empty:
            return {"net": 0.0, "campaigns": 0, "pilots": len(pilots)}
        for col in FILTER_COLUMNS + ["explicit_ids"]:
            if col not in all_campaigns.columns:
                all_campaigns[col] = None

        res = score_campaigns(all_campaigns, env.customer_profile, model, env.tariffs,
                              env.customer_profile["predicted_arpu"].sum(),
                              _mock_fallback, team_id="scenario")
    return {"net": float(res["net_arpu_gain"]), "campaigns": len(final),
            "pilots": len(pilots)}


# Миры: (название, эффект по сегментам, чего ждём от агента)
WORLDS = [
    ("как в истории", {"LOW": 1.5, "MID": 0.15, "HIGH": -0.12}, "плюс"),
    ("ПЕРЕВЁРНУТЫЙ: растёт HIGH", {"LOW": -0.3, "MID": -0.1, "HIGH": 0.4}, "плюс"),
    ("растёт только MID", {"LOW": -0.2, "MID": 0.5, "HIGH": -0.2}, "плюс"),
    ("всё слабо растёт", {"LOW": 0.05, "MID": 0.05, "HIGH": 0.05}, "плюс"),
    ("ВСЁ УБЫТОЧНО", {"LOW": -0.4, "MID": -0.3, "HIGH": -0.5}, "около нуля"),
    ("ноль везде", {"LOW": 0.0, "MID": 0.0, "HIGH": 0.0}, "около нуля"),
]


def main() -> int:
    dict_tariff = pd.read_csv("data/dict_tariff.csv")
    rng = np.random.default_rng(7)
    baseline = pd.read_csv("customer_profile.csv")["predicted_arpu"].sum()

    print(f"Baseline без действий: {baseline:,.0f}\n")
    print(f"{'мир':<28} {'ожидаем':<12} {'чистый результат':>18} {'кампаний':>9} {'пилотов':>8}  итог")
    print("-" * 94)

    failures = 0
    for name, effects, expect in WORLDS:
        model = build_model(dict_tariff, effects, rng=rng, noise=0.05)
        nets = [run_world(model, seed=s) for s in (0, 1)]
        net = sum(r["net"] for r in nets) / len(nets)
        camps = nets[0]["campaigns"]
        pilots = nets[0]["pilots"]

        # Порог убытка: 0.5% от baseline. Меньше — считаем, что агент
        # удержался; в убыточном мире идеальный ноль недостижим, потому что
        # пилоты идут в зачёт и сами стоят денег.
        limit = 0.005 * baseline
        ok = net > 0 if expect == "плюс" else net > -limit
        failures += 0 if ok else 1
        print(f"{name:<28} {expect:<12} {net:>18,.0f} {camps:>9} {pilots:>8}  "
              f"{'ok' if ok else 'ПРОВАЛ'}")

    print("-" * 94)
    print("Все миры пройдены" if not failures else f"Провалов: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
