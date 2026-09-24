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
                noise: float = 0.0,
                overrides: dict[tuple[str, str, str], float] | None = None) -> pd.DataFrame:
    """Собирает модель эффектов с заданным приростом по сегментам.

    `overrides` задаёт эффект для конкретной тройки (откуда, куда, сегмент)
    поверх сегментного — так строятся миры, где разным источникам выгодны
    разные цели.
    """
    tariffs = list(dict_tariff["tariff_plan_code"])
    rows = []
    for src in tariffs:
        for dst in tariffs:
            if src == dst:
                continue
            for seg in SEGMENTS:
                base = effect_by_segment.get(seg, 0.0)
                if overrides and (src, dst, seg) in overrides:
                    base = overrides[(src, dst, seg)]
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

    crashed: str | None = None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            final = A.Agent().act(env)
        except Exception as exc:   # noqa: BLE001 — падение агента тоже результат теста
            # Пустой план после исключения неотличим в таблице от «осторожного
            # решения ничего не делать» и может дать хороший ноль. Поэтому факт
            # падения несём отдельным полем, а не растворяем в числе.
            crashed, final = type(exc).__name__, []
        final = sanitize_campaigns(final, env.tariffs)[:MAX_CAMPAIGNS]
        pilots = internals.executed_pilot_campaigns()

        all_campaigns = pd.DataFrame(pilots + final)
        if all_campaigns.empty:
            return {"net": 0.0, "campaigns": 0, "pilots": len(pilots), "crashed": crashed}
        for col in FILTER_COLUMNS + ["explicit_ids"]:
            if col not in all_campaigns.columns:
                all_campaigns[col] = None

        res = score_campaigns(all_campaigns, env.customer_profile, model, env.tariffs,
                              env.customer_profile["predicted_arpu"].sum(),
                              _mock_fallback, team_id="scenario")
    return {"net": float(res["net_arpu_gain"]), "campaigns": len(final),
            "pilots": len(pilots), "crashed": crashed}


# Миры: (название, эффект по сегментам, чего ждём от агента)
WORLDS = [
    ("как в истории", {"LOW": 1.5, "MID": 0.15, "HIGH": -0.12}, "плюс"),
    ("ПЕРЕВЁРНУТЫЙ: растёт HIGH", {"LOW": -0.3, "MID": -0.1, "HIGH": 0.4}, "плюс"),
    ("растёт только MID", {"LOW": -0.2, "MID": 0.5, "HIGH": -0.2}, "плюс"),
    ("всё слабо растёт", {"LOW": 0.05, "MID": 0.05, "HIGH": 0.05}, "плюс"),
    ("ВСЁ УБЫТОЧНО", {"LOW": -0.4, "MID": -0.3, "HIGH": -0.5}, "около нуля"),
    ("ноль везде", {"LOW": 0.0, "MID": 0.0, "HIGH": 0.0}, "около нуля"),
]

# Мир из внешнего ревью: разным источникам выгодны РАЗНЫЕ цели. Абонентам
# MID на tariff_8 помогает только tariff_10, остальным MID — только tariff_8,
# всё прочее убыточно. Агент, навязывающий одну цель на сегмент, здесь
# отправлял 4 307 клиентов в убыточный для них тариф и терял 10,6 млн.
def _split_targets_world(dict_tariff: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    ov: dict[tuple[str, str, str], float] = {}
    for src in dict_tariff["tariff_plan_code"]:
        for dst in dict_tariff["tariff_plan_code"]:
            if src == dst:
                continue
            if src == "tariff_8" and dst == "tariff_10":
                ov[(src, dst, "MID")] = 0.6
            elif src != "tariff_8" and dst == "tariff_8":
                ov[(src, dst, "MID")] = 0.6
            else:
                ov[(src, dst, "MID")] = -0.4
    return ov


SEEDS = (0, 1)


def main() -> int:
    dict_tariff = pd.read_csv("data/dict_tariff.csv")
    rng = np.random.default_rng(7)
    baseline = pd.read_csv("customer_profile.csv")["predicted_arpu"].sum()

    # Порог убытка: 0.5% от baseline. В убыточном мире идеальный ноль
    # недостижим — пилоты идут в зачёт и сами стоят денег. Печатаем допуск
    # числом: иначе «Все миры пройдены» звучит лучше, чем есть на самом деле.
    limit = 0.005 * baseline

    print(f"Baseline без действий: {baseline:,.0f}")
    print(f"Допуск для миров «около нуля»: {-limit:,.0f} "
          f"(0,5% baseline) — убыток до этой величины считается успехом\n")
    print(f"{'мир':<28} {'ожидаем':<12} {'среднее по seed':>18} {'кампаний':>9} {'пилотов':>8}  итог")
    print("-" * 94)

    failures = 0
    worlds = list(WORLDS) + [
        ("РАЗНЫЕ ЦЕЛИ у источников", {"LOW": -0.3, "MID": -0.4, "HIGH": -0.3}, "плюс"),
    ]
    for name, effects, expect in worlds:
        overrides = _split_targets_world(dict_tariff) if "РАЗНЫЕ ЦЕЛИ" in name else None
        # Мир «ноль везде» должен быть строго нулевым: с шумом эффектов в нём
        # остаются прибыльные клетки, и проверка «удержался от действий»
        # перестаёт проверять то, что заявлено. Остальным мирам шум нужен.
        noise = 0.0 if name == "ноль везде" else 0.05
        model = build_model(dict_tariff, effects, rng=rng, noise=noise, overrides=overrides)
        runs = [run_world(model, seed=s) for s in SEEDS]
        net = sum(r["net"] for r in runs) / len(runs)
        crashed = [r for r in runs if r["crashed"]]

        # Падение агента — провал независимо от числа: пустой план после
        # исключения может дать безобидный результат и «пройти» порог.
        ok = (net > 0 if expect == "плюс" else net > -limit) and not crashed
        failures += 0 if ok else 1
        print(f"{name:<28} {expect:<12} {net:>18,.0f} {runs[0]['campaigns']:>9} "
              f"{runs[0]['pilots']:>8}  {'ok' if ok else 'ПРОВАЛ'}")
        # Два seed на мир — мало, поэтому разброс между ними показываем целиком,
        # а не прячем в среднем.
        for seed, r in zip(SEEDS, runs):
            mark = f"  ПАДЕНИЕ: {r['crashed']}" if r["crashed"] else ""
            print(f"{'':<28} {'seed ' + str(seed):<12} {r['net']:>18,.0f} "
                  f"{r['campaigns']:>9} {r['pilots']:>8}{mark}")

    print("-" * 94)
    print("Все миры пройдены" if not failures else f"Провалов: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
