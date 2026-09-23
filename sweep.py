"""Подбор параметров стратегии по медиане, а не по одному прогону.

    python sweep.py           # базовый набор конфигураций
    python sweep.py --runs 8  # больше прогонов на конфигурацию

Один прогон мок-среды — это одна случайная реализация шума пилотов. Решение,
принятое по одному числу, скорее всего окажется подгонкой под seed. Поэтому
каждая конфигурация гоняется на нескольких seed, а сравниваются медиана и
минимум: нам важно не лучшее значение, а худшее, которое мы готовы принять.
"""
from __future__ import annotations

import contextlib
import io
import statistics
import sys

import agent as A
from local_eval import evaluate_agent

# (размер пилота, запас на шум, перенос вывода на сегмент)
CONFIGS = [
    (200, 1.0, True),
    (200, 0.5, True),
    (200, 1.5, True),
    (150, 1.0, True),
    (100, 1.0, True),
    (200, 1.0, False),
]


def run(pilot_size: int, z: float, transfer: bool, seeds: range) -> list[float]:
    A.PILOT_SIZE, A.CONFIDENCE_Z, A.TRANSFER_TO_SEGMENT = pilot_size, z, transfer
    nets = []
    for seed in seeds:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):          # прогон шумный, вывод глушим
            res = evaluate_agent(A.Agent(), seed=seed, verbose=False)
        nets.append(float(res["net_arpu_gain"]))
    return nets


def main() -> int:
    runs = 5
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])
    seeds = range(runs)

    print(f"Прогонов на конфигурацию: {runs}\n")
    print(f"{'пилот':>6} {'запас':>6} {'перенос':>8} {'медиана':>14} {'минимум':>14} {'в плюс':>8}")
    print("-" * 62)

    rows = []
    for pilot_size, z, transfer in CONFIGS:
        nets = run(pilot_size, z, transfer, seeds)
        med, low = statistics.median(nets), min(nets)
        pos = sum(1 for n in nets if n > 0)
        rows.append((med, low, pos, pilot_size, z, transfer))
        print(f"{pilot_size:>6} {z:>6.1f} {str(transfer):>8} {med:>14,.0f} {low:>14,.0f} {pos:>4}/{runs}")

    best = max(rows, key=lambda r: (r[2], r[0]))       # сначала надёжность, потом медиана
    print("-" * 62)
    print(f"Лучшая конфигурация: пилот {best[3]}, запас {best[4]}, перенос {best[5]} "
          f"— медиана {best[0]:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
