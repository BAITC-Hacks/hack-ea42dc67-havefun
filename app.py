"""Веб-интерфейс агента: запуск, разбор решений, метрики.

    python app.py            # http://localhost:8000
    python app.py --port 9000

Интерфейс — надстройка над той же логикой, что и в CLI. Он ничего не считает
сам: запускает `Agent.act` на мок-среде и показывает, что агент измерил и
почему принял такое решение. Основной сценарий сдачи (`local_eval.py`,
`make_submission.py`) работает без веба и без интернета.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

import agent as A
from local_eval import evaluate_agent

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

# Данные читаются по путям относительно корня проекта, поэтому сервер должен
# работать из него независимо от того, откуда его запустили.
os.chdir(ROOT)

app = FastAPI(title="Агент тарифных кампаний", docs_url="/api/docs")


class RunRequest(BaseModel):
    seed: int = 42
    pilot_size: int = A.PILOT_SIZE
    confidence_z: float = A.CONFIDENCE_Z
    transfer: bool = True


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "pilot_limit": A.N_PILOTS, "campaign_limit": A.MAX_CAMPAIGNS}


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    """Прогон агента на мок-среде с заданными параметрами."""
    A.PILOT_SIZE = max(10, min(200, req.pilot_size))
    A.CONFIDENCE_Z = req.confidence_z
    A.TRANSFER_TO_SEGMENT = req.transfer

    started = time.time()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = evaluate_agent(A.Agent(), seed=req.seed, verbose=False)
    elapsed = time.time() - started

    text = log.getvalue()
    explanation = ""
    if "=== Обоснование плана ===" in text:
        explanation = text.split("=== Обоснование плана ===")[1]
        explanation = explanation.split("=== конец обоснования ===")[0].strip()

    campaigns, pilots = [], []
    for c in result.get("campaigns_detail", []):
        row = {
            "name": c.get("name", ""),
            "channel": c.get("channel", ""),
            "contacts": int(c.get("n_contacts", 0)),
            "cost": float(c.get("cost", 0.0)),
            "gross_lift": float(c.get("gross_lift", 0.0)),
            "negative": int(c.get("n_negative", 0)),
            "capped": bool(c.get("capped_at_campaign_limit") or c.get("capped_at_reach_budget") or c.get("capped_at_money_budget")),
        }
        (pilots if row["name"].startswith("pilot_") else campaigns).append(row)

    return {
        "seed": req.seed,
        "elapsed_s": round(elapsed, 2),
        "status": result.get("status", ""),
        "baseline": float(result.get("baseline_total_arpu", 0.0)),
        "gross": float(result.get("gross_arpu_lift", 0.0)),
        "cost": float(result.get("total_cost", 0.0)),
        "net": float(result.get("net_arpu_gain", 0.0)),
        "growth_pct": float(result.get("growth_vs_baseline_pct", 0.0)),
        "contacts": int(result.get("total_contacts", 0)),
        "unique_customers": int(result.get("unique_customers_targeted", 0)),
        "coverage_pct": float(result.get("coverage_pct", 0.0)),
        "risk_pct": float(result.get("risk_score_pct", 0.0)),
        "budget_used_pct": float(result.get("budget_used_pct", 0.0)),
        "n_pilots": int(result.get("n_pilots", 0)),
        "campaigns": campaigns,
        "pilots": pilots,
        "explanation": explanation,
    }


class AskRequest(BaseModel):
    question: str
    seed: int = 42


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """Вопрос аналитика к готовому плану на естественном языке.

    Модель получает ТОЛЬКО посчитанные числа: план, результаты пилотов,
    статистику по сегментам. Она ничего не пересчитывает и не решает —
    отвечает по фактам. Если ключа нет, честно говорит об этом.
    """
    question = (req.question or "").strip()
    if not question:
        return {"answer": "Вопрос пустой.", "model": None}
    if len(question) > 500:
        question = question[:500]

    data = run(RunRequest(seed=req.seed))
    facts = [
        f"Baseline без кампаний: {data['baseline']:,.0f} у.е.",
        f"Чистый результат плана: {data['net']:,.0f} у.е. ({data['growth_pct']:.2f}% к baseline).",
        f"Затраты на коммуникацию: {data['cost']:,.0f} у.е. из 100 000 бюджета.",
        f"Охват: {data['unique_customers']:,} абонентов из 23 441, контактов {data['contacts']:,} из 15 000.",
        f"Проведено пилотов: {data['n_pilots']} из 20.",
        "Кампании плана:",
    ]
    for c in data["campaigns"]:
        facts.append(
            f"  — {c['name']}: канал {c['channel']}, {c['contacts']:,} контактов, "
            f"затраты {c['cost']:,.0f}, прирост {c['gross_lift']:,.0f} у.е.")
    seg = insights()["segments"]
    facts.append("Медианный прирост ARPU по сегментам в истории: " + ", ".join(
        f"{x['segment']} {x['median_lift']:+.2f} ({x['share_pct']}% базы)" for x in seg))
    facts.append("Сегменты с отрицательным приором в кампании не берутся: "
                 "деньги на них теряются, а не зарабатываются.")
    facts.append("Каналы: push 0 у.е., sms 4, реклама 22, звонок 160 за контакт. "
                 "Множитель конверсии соответственно 0.50, 0.65, 0.85, 1.20, "
                 "но произведение ограничено сверху единицей.")

    try:
        from openai import OpenAI

        import explain as E

        if not os.environ.get("OPENAI_API_KEY"):
            return {"answer": "Ключ OPENAI_API_KEY не задан, поэтому отвечать некому. "
                              "План и все числа при этом посчитаны и показаны выше — "
                              "модель на них не влияет.", "model": None}
        client = OpenAI(timeout=30)
        model = E._pick_model(client)
        if model is None:
            return {"answer": "Доступной модели не нашлось на этом ключе.", "model": None}
        user_content = ("Факты:" + chr(10) + chr(10).join(facts)
                        + chr(10) + chr(10) + "Вопрос: " + question)
        system = ("Ты помогаешь аналитику маркетинга разобраться в плане кампаний. "
                  "Отвечай ТОЛЬКО по переданным фактам, не выдумывай числа и не "
                  "пересчитывай их. Если ответа в фактах нет, так и скажи. "
                  "Коротко, по-русски, без маркетингового пафоса.")
        return {"answer": E.call_model(client, model, system, user_content), "model": model}
    except Exception as exc:  # noqa: BLE001 — вопрос не должен ронять сервис
        return {"answer": f"Модель недоступна ({type(exc).__name__}). "
                          f"План и числа при этом посчитаны и не зависят от неё.",
                "model": None}


@app.get("/api/insights")
def insights() -> dict:
    """Разбор данных: на чём строится приор и почему HIGH исключён.

    Считается из той же истории, что использует агент, — чтобы в интерфейсе
    были ровно те числа, на которых принимается решение.
    """
    import numpy as np
    import pandas as pd

    prof = pd.read_csv("customer_profile.csv")
    hist = pd.read_csv(A.HISTORY_PATH)
    hist = hist[hist["AVG_ARPU_PREV_3M"] > 0].copy()
    hist["rel"] = ((hist["AVG_ARPU_NEXT_3M"] - hist["AVG_ARPU_PREV_3M"])
                   / hist["AVG_ARPU_PREV_3M"])
    hist["arpu_segment"] = pd.cut(hist["AVG_ARPU_PREV_3M"], [-np.inf, 1000, 5000, np.inf],
                                  labels=["LOW", "MID", "HIGH"])

    seg_stats = []
    sizes = prof["arpu_segment"].value_counts()
    total = int(len(prof))
    for seg in ["LOW", "MID", "HIGH"]:
        rows = hist[hist["arpu_segment"] == seg]
        n = int(sizes.get(seg, 0))
        seg_stats.append({
            "segment": seg,
            "median_lift": float(rows["rel"].median()) if len(rows) else 0.0,
            "customers": n,
            "share_pct": round(100 * n / total, 1) if total else 0.0,
            "history_rows": int(len(rows)),
            "mean_arpu": float(prof.loc[prof["arpu_segment"] == seg, "predicted_arpu"].mean() or 0),
        })

    targets = []
    for seg in ["LOW", "MID"]:
        rows = hist[hist["arpu_segment"] == seg]
        g = (rows.groupby("tariff_plan_code_to")["rel"]
                 .agg(["median", "size"]).reset_index())
        g = g[g["size"] >= A.MIN_HISTORY_ROWS].nlargest(4, "median")
        for _, r in g.iterrows():
            targets.append({
                "segment": seg,
                "target": r["tariff_plan_code_to"],
                "median_lift": float(r["median"]),
                "n": int(r["size"]),
            })

    return {"segments": seg_stats, "targets": targets, "total_customers": total}


@app.get("/api/compare")
def compare() -> dict:
    """Сравнение двух стратегий отбора на одних и тех же seed.

    Показывает цену главного решения в архитектуре: переносить ли
    подтверждённый вывод на непилотированные исходные тарифы сегмента.
    """
    out = {}
    for key, transfer in (("transfer", True), ("measured_only", False)):
        A.TRANSFER_TO_SEGMENT = transfer
        nets, reach = [], []
        for seed in range(5):
            with contextlib.redirect_stdout(io.StringIO()):
                res = evaluate_agent(A.Agent(), seed=seed, verbose=False)
            nets.append(float(res["net_arpu_gain"]))
            reach.append(int(res["unique_customers_targeted"]))
        ordered = sorted(nets)
        out[key] = {
            "median": ordered[len(ordered) // 2],
            "min": min(nets),
            "positive": sum(1 for n in nets if n > 0),
            "avg_reach": round(sum(reach) / len(reach)),
        }
    A.TRANSFER_TO_SEGMENT = True
    gain = out["transfer"]["median"] - out["measured_only"]["median"]
    out["delta_pct"] = round(100 * gain / abs(out["measured_only"]["median"] or 1), 1)
    return out


@app.get("/api/scenarios")
def scenarios_api() -> dict:
    """Прогон агента в мирах с другой моделью эффектов.

    Прямая проверка предупреждения из ТЗ: на судействе эффекты другие.
    Считается тем же кодом, что и `python scenarios.py`.
    """
    import pandas as pd

    import scenarios as S

    dict_tariff = pd.read_csv("data/dict_tariff.csv")
    baseline = float(pd.read_csv("customer_profile.csv")["predicted_arpu"].sum())
    rng = np.random.default_rng(7)
    limit = 0.005 * baseline

    worlds = []
    for name, effects, expect in S.WORLDS:
        model = S.build_model(dict_tariff, effects, rng=rng, noise=0.05)
        runs = [S.run_world(model, seed=seed) for seed in (0, 1)]
        net = sum(r["net"] for r in runs) / len(runs)
        worlds.append({
            "name": name,
            "expect": expect,
            "net": net,
            "campaigns": runs[0]["campaigns"],
            "pilots": runs[0]["pilots"],
            "ok": bool(net > 0) if expect == "плюс" else bool(net > -limit),
            "effects": effects,
        })
    return {"baseline": baseline, "worlds": worlds,
            "passed": sum(1 for w in worlds if w["ok"]), "total": len(worlds)}


@app.get("/api/whatif")
def whatif(budget: int = 50000) -> dict:
    """Пересчёт плана при другом бюджете на коммуникацию.

    Аналитик спрашивает «а если денег дадут вдвое меньше» — и это
    считается, а не обсуждается. Агент заново проводит разведку и
    перераспределяет каналы под новое ограничение.
    """
    import pandas as pd

    from environment import make_environment
    from mock_environment import (CHANNELS as ENV_CHANNELS, MAX_TOTAL_CONTACTS,
                                  _mock_fallback, _mock_impact_model)
    from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns

    budget = max(1000, min(100_000, int(budget)))
    change_tariff = pd.read_csv("data/change_tariff.csv")
    model = _mock_impact_model(change_tariff)
    profile = pd.read_csv("customer_profile.csv")
    dict_tariff = pd.read_csv("data/dict_tariff.csv")

    env, internals = make_environment(
        customer_profile=profile, impact_model=model, dict_tariff=dict_tariff,
        channels=ENV_CHANNELS, total_budget=budget,
        max_total_contacts=MAX_TOTAL_CONTACTS, fallback_predict=_mock_fallback, seed=42)

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            final = A.Agent().act(env)
        except Exception:  # noqa: BLE001 — пустой план тоже ответ
            final = []
        final = sanitize_campaigns(final, env.tariffs)[:MAX_CAMPAIGNS]
        pilots = internals.executed_pilot_campaigns()
        rows = pd.DataFrame(pilots + final)
        if rows.empty:
            return {"budget": budget, "net": 0.0, "contacts": 0, "campaigns": 0, "channels": {}}
        for col in ["filter_arpu_segment", "filter_data_segment", "filter_call_segment",
                    "filter_current_tariff", "explicit_ids"]:
            if col not in rows.columns:
                rows[col] = None
        res = score_campaigns(rows, env.customer_profile, model, env.tariffs,
                              float(env.customer_profile["predicted_arpu"].sum()),
                              _mock_fallback, team_id="whatif")

    channels: dict[str, int] = {}
    for c in final:
        channels[c["channel"]] = channels.get(c["channel"], 0) + 1
    return {
        "budget": budget,
        "net": float(res["net_arpu_gain"]),
        "cost": float(res["total_cost"]),
        "contacts": int(res["total_contacts"]),
        "customers": int(res["unique_customers_targeted"]),
        "campaigns": len(final),
        "channels": channels,
    }


@app.get("/api/stability")
def stability(runs: int = 5) -> dict:
    """Прогон на нескольких seed — устойчивость знака результата."""
    runs = max(2, min(10, runs))
    nets = []
    for seed in range(runs):
        with contextlib.redirect_stdout(io.StringIO()):
            res = evaluate_agent(A.Agent(), seed=seed, verbose=False)
        nets.append(float(res["net_arpu_gain"]))
    ordered = sorted(nets)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "runs": runs,
        "nets": nets,
        "median": median,
        "min": min(nets),
        "max": max(nets),
        "positive": sum(1 for n in nets if n > 0),
    }


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")


def main() -> int:
    import uvicorn

    port = 8000
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"Интерфейс: http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
