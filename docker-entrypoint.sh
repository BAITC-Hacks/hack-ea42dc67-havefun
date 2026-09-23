#!/bin/sh
# Точка входа контейнера: один аргумент выбирает сценарий.
set -e

case "${1:-run}" in
  run)        exec python local_eval.py ;;
  selfcheck)  exec python selfcheck.py ;;
  stability)  exec python local_eval.py --runs 5 ;;
  sweep)      exec python sweep.py --runs 5 ;;
  scenarios)  exec python scenarios.py ;;
  submission) exec python make_submission.py ;;
  web)        exec python -m uvicorn app:app --host 0.0.0.0 --port 8000 ;;
  *)          exec "$@" ;;
esac
