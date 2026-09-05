#!/bin/sh
# Старт сервиса. Веса модели не лежат в образе: их 110 МБ, они не меняются
# и от версии кода не зависят — поэтому качаются один раз в volume, а не
# пересобираются вместе с образом.
#
# DOWNLOAD_WEIGHTS=0 отключает загрузку: на закрытом контуре файл кладут в
# volume руками.
set -e

if [ "${MODEL_BACKEND}" = "segformer" ] && [ "${DOWNLOAD_WEIGHTS:-1}" = "1" ]; then
    if [ ! -f "${MODEL_WEIGHTS}" ]; then
        echo "веса не найдены в ${MODEL_WEIGHTS} — качаю"
        python scripts/download_model.py --out "${MODEL_WEIGHTS}"
    fi
fi

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8001}" \
    --workers "${WORKERS:-1}"
