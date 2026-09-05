# syntax=docker/dockerfile:1
#
# Два целевых образа, и это главное решение файла:
#
#   --target runtime  (по умолчанию) — сервис + правиловый бэйзлайн. Десятки
#       мегабайт зависимостей, сборка меньше минуты. Этого хватает, чтобы
#       поднять API, отдать контракт фронту и бэку и прогнать тесты.
#   --target model — то же плюс torch и transformers, ~1.5 ГБ. Собирается
#       долго, поэтому вынесено в отдельный слой поверх готового runtime:
#       правка кода сервиса не заставляет переустанавливать torch.
#
# Что ещё сделано ради скорости сборки:
#   * зависимости ставятся до копирования кода — правка `app/` не сбрасывает
#     кеш установки;
#   * тяжёлый слой torch стоит последним и в отдельном таргете;
#   * кеш pip прокинут через `--mount=type=cache`, поэтому повторная сборка
#     не качает колёса заново (нужен BuildKit: DOCKER_BUILDKIT=1);
#   * веса (110 МБ) в образ не кладутся вообще — они приезжают в volume при
#     первом старте, иначе каждый пересбор образа тащил бы их с собой.

FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# runtime — сервис без обученной модели
# ---------------------------------------------------------------------------
FROM base AS runtime

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

ENV MODEL_BACKEND=heuristic \
    MODEL_WEIGHTS=/models/model.pt \
    PYTHONPATH=/app

RUN useradd --system --create-home appuser \
    && mkdir -p /models \
    && chown appuser:appuser /models
USER appuser

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

ENTRYPOINT ["./docker-entrypoint.sh"]

# ---------------------------------------------------------------------------
# model — runtime + партнёрская модель «Чистый берег»
# ---------------------------------------------------------------------------
FROM runtime AS model
USER root

COPY requirements-model.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-model.txt

USER appuser
ENV MODEL_BACKEND=segformer
