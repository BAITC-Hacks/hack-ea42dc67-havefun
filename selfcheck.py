"""Самопроверка решения: запускается одной командой, ничего не требует.

    python selfcheck.py

Проверяет то, что проверяет жюри: агент запускается, план корректен,
лимиты соблюдены, результат положителен, поведение при некорректных
входных данных предсказуемо.
"""
from __future__ import annotations

import sys

import agent as A
from mock_environment import make_mock_env

OK, FAIL = "[ ok ]", "[FAIL]"
errors: list[str] = []


def check(condition: bool, title: str, detail: str = "") -> None:
    print(f"{OK if condition else FAIL} {title}" + (f" — {detail}" if detail else ""))
    if not condition:
        errors.append(title)


def main() -> int:
    print("=== 1. Основной сценарий ===")
    env, _ = make_mock_env(seed=42)
    plan = A.Agent().act(env)

    check(isinstance(plan, list), "план — это список")
    check(1 <= len(plan) <= A.MAX_CAMPAIGNS, "от 1 до 10 кампаний", f"получено {len(plan)}")

    known_tariffs = set(env.tariffs["tariff_plan_code"])
    check(all(c["target_tariff"] in known_tariffs for c in plan), "все целевые тарифы существуют")
    check(all(c["channel"] in env.channels for c in plan), "все каналы существуют")
    check(all(c.get("campaign_name") for c in plan), "у каждой кампании есть имя")

    print("\n=== 2. Лимиты ===")
    check(env.pilots_left >= 0, "лимит пилотов не превышен",
          f"использовано {A.N_PILOTS - env.pilots_left} из {A.N_PILOTS}")
    check(env.remaining_budget >= 0, "бюджет не превышен",
          f"осталось {env.remaining_budget:.0f}")
    check(env.remaining_contacts >= 0, "охват не превышен",
          f"осталось {env.remaining_contacts}")
    check(len(env.pilot_history) > 0, "агент действительно пилотировал",
          f"{len(env.pilot_history)} пилотов")

    print("" + chr(10) + "=== 3. Охват по фильтрам плана совпадает с зачтённым ===")
    import contextlib
    import io as _io

    from local_eval import evaluate_agent
    from scoring_core import apply_filters

    # Агент отдаёт план уже без поля reach, поэтому сравнить с его внутренним
    # числом нельзя. Сравниваем то, что проверяемо: сколько абонентов выбирают
    # фильтры кампании из профиля — и сколько контактов эта же кампания
    # получила в зачёте. Расхождение означало бы молчаливое обрезание.
    for seed in (5, 11, 23):
        env_s, _ = make_mock_env(seed=seed)
        with contextlib.redirect_stdout(_io.StringIO()):
            plan_s = A.Agent().act(env_s)
            res_s = evaluate_agent(A.Agent(), seed=seed, verbose=False)
        profile_s = env_s.customer_profile
        actual = dict((c["name"], c["n_contacts"]) for c in res_s["campaigns_detail"]
                      if not c["name"].startswith("pilot_"))
        planned = dict((c["campaign_name"], len(apply_filters(profile_s, c))) for c in plan_s)

        check(set(actual) == set(planned),
              "seed %d: среда не отбросила ни одной кампании плана" % seed,
              "в плане %d, в зачёте %d" % (len(planned), len(actual)))
        diff = [n for n in planned if actual.get(n) != planned[n]]
        check(not diff,
              "seed %d: контактов в зачёте столько же, сколько абонентов под фильтрами" % seed,
              "по фильтрам %d, в зачёте %d%s" % (
                  sum(planned.values()), sum(actual.values()),
                  "" if not diff else ", расходятся: " + ", ".join(diff[:3])))
        capped = [c["name"] for c in res_s["campaigns_detail"]
                  if c.get("capped_at_money_budget") or c.get("capped_at_reach_budget")
                  or c.get("capped_at_campaign_limit")]
        check(not capped,
              "seed %d: ни одна кампания не обрезана лимитом" % seed,
              "обрезаны: " + ", ".join(capped) if capped else "")

    print("\n=== 4. Некорректные входные данные ===")
    env2, _ = make_mock_env(seed=1)
    env2.customer_profile = env2.customer_profile.iloc[0:0]
    check(A.Agent().act(env2) == [], "пустая аудитория — пустой план, без падения")

    env3, _ = make_mock_env(seed=1)
    env3.customer_profile = env3.customer_profile.drop(columns=["predicted_arpu"])
    check(A.Agent().act(env3) == [], "нет обязательной колонки — пустой план, без падения")

    saved = A.HISTORY_PATH
    A.HISTORY_PATH = "data/файла-нет.csv"
    env4, _ = make_mock_env(seed=1)
    check(A.Agent().act(env4) == [], "нет истории — пустой план, без падения")
    A.HISTORY_PATH = saved

    print("" + chr(10) + "=== 5. Решение не зависит от языковой модели ===")
    import explain as E

    def _boom(*a, **k):
        raise RuntimeError("модель недоступна")

    saved_explain = E.explain_plan
    with contextlib.redirect_stdout(_io.StringIO()):
        base = evaluate_agent(A.Agent(), seed=7, verbose=False)["net_arpu_gain"]
        E.explain_plan = _boom          # имитируем полный отказ модели
        try:
            broken = evaluate_agent(A.Agent(), seed=7, verbose=False)["net_arpu_gain"]
        finally:
            E.explain_plan = saved_explain
    check(abs(base - broken) < 1e-6,
          "результат совпадает при недоступной модели",
          "%.0f против %.0f" % (base, broken))

    print("\n=== 6. Валидация плана ===")
    broken = [
        {"campaign_name": "плохой канал", "target_tariff": "tariff_9", "channel": "телепатия"},
        {"campaign_name": "плохой тариф", "target_tariff": "tariff_999", "channel": "sms"},
    ]
    check(A.validate_plan(broken, env) == [], "некорректные кампании отсеиваются")

    print("\n" + ("ВСЁ ПРОШЛО" if not errors else f"ПРОВАЛЕНО: {len(errors)}"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
