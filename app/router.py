"""
HTTP-слой. Тонкий: разбирает запрос, зовёт `pipeline.run`, кладёт артефакты.

Ручка одна и синхронная. Инференс — CPU/GPU-bound, `async def` его не ускорит,
а вот event loop заблокирует; FastAPI сам уводит `def`-эндпоинты в пул потоков.
"""

from __future__ import annotations

import io
import json
import logging
import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from app import estimate, pipeline, taxonomy, tiles
from app.backends import get_detector
from app.config import settings
from app.schemas import (
    AreaRequest,
    ClassInfoOut,
    DetectOptions,
    DetectResponse,
    FraudFlag,
    HealthResponse,
    ImageryInfo,
    ModelInfo,
    ModelResponse,
    OverlayOut,
    TileSourceOut,
)
from app.store import ArtifactStore

log = logging.getLogger("ml.router")

router = APIRouter()

artifacts = ArtifactStore(
    ttl_seconds=settings.ARTIFACT_TTL_SECONDS,
    max_items=settings.ARTIFACT_MAX_ITEMS,
)


@lru_cache(maxsize=1)
def available_sources() -> dict[str, tiles.TileSource]:
    """Источники по умолчанию плюс подключённые конфигом.

    Своя ортофото-мозаика ООПТ добавляется сюда переменной `TILE_SOURCES` —
    именно она даёт сантиметровое разрешение, которого нет у публичных
    подложек, и ради неё весь тайловый путь и сделан.
    """
    sources = dict(tiles.DEFAULT_SOURCES)
    for name, spec in settings.TILE_SOURCES.items():
        sources[name] = tiles.TileSource(name=name, **spec)
    return sources


@lru_cache(maxsize=1)
def _fetcher() -> tiles.TileFetcher:
    cache_dir = Path(settings.TILE_CACHE_DIR) if settings.TILE_CACHE_DIR else None
    return tiles.TileFetcher(cache_dir=cache_dir, workers=settings.TILE_WORKERS)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Общий секрет бэкенд->ML. Пустой ключ в конфиге = проверка выключена."""
    if not settings.API_KEY:
        return
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _model_info() -> ModelInfo:
    info = get_detector(settings.MODEL_BACKEND).info()
    return ModelInfo(
        backend=info.name,
        version=info.version,
        weights=info.weights,
        device=info.device,
        trained=info.trained,
        notes=info.notes,
    )


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Готовность. Модель грузится лениво, поэтому её отказ здесь не 500."""
    try:
        info = get_detector(settings.MODEL_BACKEND).info()
    except Exception as exc:  # noqa: BLE001 — health не должен падать
        log.warning("backend %s не поднялся: %s", settings.MODEL_BACKEND, exc)
        return HealthResponse(
            status="ok",
            backend=settings.MODEL_BACKEND,
            backend_ready=False,
            trained=False,
            version="unknown",
        )
    return HealthResponse(
        status="ok",
        backend=info.name,
        backend_ready=True,
        trained=info.trained,
        version=info.version,
    )


@router.get("/api/v1/model", response_model=ModelResponse, tags=["service"])
def model_card() -> ModelResponse:
    """Паспорт модели: кто считает, какие классы и на каких допущениях."""
    return ModelResponse(
        model=_model_info(),
        classes=[
            ClassInfoOut(
                id=c.id,
                label=c.label,
                label_ru=c.label_ru,
                trash_category=c.trash_category,
                color_hex=c.color_hex,
            )
            for c in taxonomy.all_classes()
        ],
        assumptions=estimate.assumptions(),
    )


@router.post(
    "/api/v1/imagery/detect",
    response_model=DetectResponse,
    tags=["imagery"],
    dependencies=[Depends(require_api_key)],
)
def detect(
    image: UploadFile = File(..., description="Кадр БПЛА или фрагмент ортофото"),
    meta: str | None = Form(
        default=None, description="JSON с полями DetectOptions: GSD, углы кадра, ООПТ"
    ),
) -> DetectResponse:
    """Найти мусор на кадре и вернуть кандидатов в точки для очереди ООПТ."""
    options = _parse_meta(meta)
    frame = _read_image(image)

    try:
        detector = get_detector(settings.MODEL_BACKEND)
    except Exception as exc:
        log.exception("не удалось поднять бэкенд %s", settings.MODEL_BACKEND)
        raise HTTPException(status_code=503, detail=f"модель недоступна: {exc}") from exc

    return _to_response(pipeline.run(frame, options, detector))


def _to_response(result: pipeline.PipelineResult) -> DetectResponse:
    """Результат конвейера -> HTTP-ответ. Общий для обеих ручек детекции."""
    overlay_url = blend_url = None
    if result.overlay_png is not None:
        artifacts.put(f"{result.job_id}:overlay", result.overlay_png, "image/png")
        overlay_url = f"/api/v1/imagery/{result.job_id}/overlay.png"
    if result.blend_png is not None:
        artifacts.put(f"{result.job_id}:blend", result.blend_png, "image/png")
        blend_url = f"/api/v1/imagery/{result.job_id}/blend.png"

    return DetectResponse(
        job_id=result.job_id,
        model=result.model,
        image=result.image,
        detections=result.detections,
        summary=result.summary,
        point_candidates=result.point_candidates,
        fraud_flags=result.fraud_flags,
        exif=result.exif,
        overlay=OverlayOut(
            png_url=overlay_url,
            blend_url=blend_url,
            bounds=result.overlay_bounds,
            accent_hex=taxonomy.to_hex(taxonomy.OVERLAY_ACCENT),
            expires_in_seconds=settings.ARTIFACT_TTL_SECONDS,
        ),
        geojson=result.geojson,
        assumptions=estimate.assumptions(),
        timing_ms=result.timing_ms,
    )


@router.get("/api/v1/tiles/sources", response_model=list[TileSourceOut], tags=["imagery"])
def tile_sources() -> list[TileSourceOut]:
    """Подключённые тайловые источники. `attribution` обязателен к показу."""
    return [
        TileSourceOut(
            name=source.name,
            attribution=source.attribution,
            max_zoom=source.max_zoom,
            tile_size=source.tile_size,
        )
        for source in available_sources().values()
    ]


@router.post(
    "/api/v1/imagery/detect/area",
    response_model=DetectResponse,
    tags=["imagery"],
    dependencies=[Depends(require_api_key)],
)
def detect_area(request: AreaRequest) -> DetectResponse:
    """Найти мусор на участке карты: сервис сам заберёт тайлы подложки.

    Привязка здесь точная — координаты тайловой сетки заданы формулой, а не
    восстановлены из EXIF. Зато разрешение задаёт провайдер, и если пиксель
    крупнее различимого моделью объекта, кандидаты в очередь ООПТ не создаются:
    см. `imagery.too_coarse` в ответе.
    """
    sources = available_sources()
    name = request.source or settings.TILE_DEFAULT_SOURCE
    source = sources.get(name)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail=f"источник {name!r} не подключён; доступны: {sorted(sources)}",
        )
    if request.zoom > source.max_zoom:
        raise HTTPException(
            status_code=422,
            detail=f"источник {name!r} отдаёт максимум зум {source.max_zoom}",
        )

    rng = tiles.tile_range(request.bbox, request.zoom)
    if rng.count > settings.TILE_MAX_PER_REQUEST:
        raise HTTPException(
            status_code=422,
            detail=(
                f"участок требует {rng.count} тайлов при пределе "
                f"{settings.TILE_MAX_PER_REQUEST} — уменьшите площадь или зум"
            ),
        )

    started = time.perf_counter()
    try:
        mosaic = tiles.build_mosaic(source, request.bbox, request.zoom, _fetcher())
    except tiles.TileError as exc:
        raise HTTPException(status_code=502, detail=f"подложка недоступна: {exc}") from exc
    fetch_ms = round((time.perf_counter() - started) * 1000, 1)

    try:
        detector = get_detector(settings.MODEL_BACKEND)
    except Exception as exc:
        log.exception("не удалось поднять бэкенд %s", settings.MODEL_BACKEND)
        raise HTTPException(status_code=503, detail=f"модель недоступна: {exc}") from exc

    suppress = mosaic.too_coarse and settings.TILE_BLOCK_COARSE_CANDIDATES
    options = DetectOptions(
        territory_id=request.territory_id,
        min_confidence=request.min_confidence,
        min_area_px=request.min_area_px,
        render_overlay=request.render_overlay,
        overlay_by_class=request.overlay_by_class,
        include_geojson=request.include_geojson,
        suppress_candidates=suppress,
    )
    result = pipeline.run(
        mosaic.image, options, detector, georeference=mosaic.georeference
    )
    result.timing_ms["tiles_fetch"] = fetch_ms

    response = _to_response(result)
    response.imagery = ImageryInfo(
        source=source.name,
        attribution=source.attribution,
        zoom=mosaic.zoom,
        tiles_total=mosaic.tiles_total,
        tiles_missing=mosaic.tiles_missing,
        gsd_m_per_px=round(mosaic.gsd_m_per_px, 4),
        too_coarse=mosaic.too_coarse,
        candidates_suppressed=suppress,
    )
    response.fraud_flags = [*response.fraud_flags, *_imagery_flags(mosaic, suppress)]
    return response


def _imagery_flags(mosaic: tiles.Mosaic, suppressed: bool) -> list[FraudFlag]:
    """Предупреждения о самой подложке — их видит модератор, а не только лог."""
    flags: list[FraudFlag] = []
    if mosaic.looks_like_placeholder:
        flags.append(
            FraudFlag(
                code="imagery_unavailable",
                severity="warning",
                message=(
                    "Провайдер вернул заглушку вместо снимка: на этом зуме "
                    "съёмки нет. Пустой результат не означает чистый берег."
                ),
                details={"zoom": mosaic.zoom, "source": mosaic.source.name},
            )
        )
    if mosaic.too_coarse:
        flags.append(
            FraudFlag(
                code="imagery_too_coarse",
                severity="warning",
                message=(
                    f"Пиксель подложки {mosaic.gsd_m_per_px:.2f} м при пределе "
                    f"{tiles.COARSE_IMAGERY_GSD_M} м. Модель обучена на съёмке БПЛА "
                    "с сантиметровым разрешением; здесь различимы только очень "
                    "крупные объекты."
                    + (" Кандидаты в точки не создавались." if suppressed else "")
                ),
                details={
                    "gsd_m_per_px": round(mosaic.gsd_m_per_px, 4),
                    "threshold_m": tiles.COARSE_IMAGERY_GSD_M,
                    "zoom": mosaic.zoom,
                },
            )
        )
    if mosaic.tiles_missing:
        flags.append(
            FraudFlag(
                code="imagery_incomplete",
                severity="info",
                message=f"Не получено тайлов: {mosaic.tiles_missing} из {mosaic.tiles_total}",
            )
        )
    return flags


@router.get("/api/v1/imagery/{job_id}/overlay.png", tags=["imagery"])
def overlay(job_id: str) -> Response:
    """PNG-оверлей: прозрачный фон, фиолетовые пятна. Живёт TTL, потом 404."""
    return _artifact(f"{job_id}:overlay")


@router.get("/api/v1/imagery/{job_id}/blend.png", tags=["imagery"])
def blend(job_id: str) -> Response:
    """Кадр со впечатанным оверлеем — для карточки модератора и отчёта."""
    return _artifact(f"{job_id}:blend")


def _artifact(key: str) -> Response:
    item = artifacts.get(key)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail="артефакт не найден или истёк — повторите запрос детекции",
        )
    return Response(
        content=item.content,
        media_type=item.media_type,
        headers={"Cache-Control": f"private, max-age={settings.ARTIFACT_TTL_SECONDS}"},
    )


def _parse_meta(meta: str | None) -> DetectOptions:
    if not meta:
        return DetectOptions()
    try:
        payload = json.loads(meta)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"meta не JSON: {exc}") from exc
    try:
        return DetectOptions.model_validate(payload)
    except ValidationError as exc:
        # Через .json(), а не .errors(): в `ctx` лежит исходное исключение
        # валидатора, и оно не сериализуется в ответ.
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


def _read_image(upload: UploadFile) -> Image.Image:
    """Читает кадр целиком в память с ограничением размера."""
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    raw = upload.file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(
            status_code=413, detail=f"кадр больше {settings.MAX_UPLOAD_MB} МБ"
        )
    if not raw:
        raise HTTPException(status_code=422, detail="пустой файл")
    try:
        frame = Image.open(io.BytesIO(raw))
        frame.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=422, detail=f"не удалось прочитать изображение: {exc}"
        ) from exc
    return frame
