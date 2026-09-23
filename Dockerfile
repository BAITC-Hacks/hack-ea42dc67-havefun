# Образ для воспроизведения решения без установки Python на машине проверяющего.
#
#   docker build -t hackalem-agent .
#   docker run --rm hackalem-agent                  # прогон агента и отчёт
#   docker run --rm hackalem-agent selfcheck        # самопроверка
#   docker run --rm hackalem-agent stability        # устойчивость по 5 seed
#   docker run --rm -p 8000:8000 hackalem-agent web # веб-интерфейс
#
# Docker — дополнение, а не замена: всё то же самое запускается из репозитория
# напрямую, см. README.

FROM python:3.12-slim

# Зависимости ставим отдельным слоем, чтобы пересборка после правки кода
# не тянула их заново.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Агент и данные читаются по путям относительно корня проекта.
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

EXPOSE 8000

# Проверка живости для режима web: контейнер считается здоровым только если
# приложение действительно отвечает, а не просто запустился процесс.
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["run"]
