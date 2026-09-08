# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBRA_DATA_DIR=/data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[server,cli]" \
    && useradd --create-home --uid 10001 embra \
    && mkdir -p /data && chown -R embra:embra /data /app

USER embra
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"

CMD ["uvicorn", "embra.server.app:create_default_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080"]
