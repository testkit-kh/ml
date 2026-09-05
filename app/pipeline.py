"""
Конвейер инференса: кадр + метаданные -> детекции, оверлей, кандидаты в точки.

Отдельно от HTTP-слоя, чтобы то же самое можно было гонять из скрипта оценки
качества на тестовых данных, не поднимая сервер.

Порядок шагов:
    кадр -> EXIF -> масштаб -> сегментация -> пятна -> замеры -> привязка
         -> сводка -> кандидаты -> антифрод -> оверлей
"""

from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from PIL import Image

from app import estimate, georef, masks, taxonomy
from app.backends.base import Detector
from app.config import settings
from app.schemas import (
    ClassSummary,
    Detection,
    DetectOptions,
    ExifOut,
    FraudFlag,
    ImageInfo,
    ModelInfo,
    PointCandidate,
    Summary,
)


@dataclass
class PipelineResult:
    job_id: str
    model: ModelInfo
    image: ImageInfo
    detections: list[Detection]
    summary: Summary
    point_candidates: list[PointCandidate]
    fraud_flags: list[FraudFlag]
    exif: ExifOut
    geojson: dict | None
    overlay_png: bytes | None
    blend_png: bytes | None
    overlay_bounds: list[list[float]] | None
    timing_ms: dict[str, float]
    label_mask: np.ndarray


def _fit_to_max_side(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Ужимает длинную сторону до `max_side`. Возвращает кадр и его масштаб.

    Инференс на кадре 8000 px — это минуты и гигабайты; при этом мусор,
    различимый на 8000, различим и на 4096. Все пиксельные координаты потом
    делятся на масштаб, поэтому клиент получает их в системе исходного кадра.
    """
    longest = max(image.size)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    size = (max(int(image.width * scale), 1), max(int(image.height * scale), 1))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def _resolve_gsd(
    options: DetectOptions,
    *,
    width_px: int,
    scale: float,
) -> tuple[float | None, str]:
    """GSD в пикселях рабочего кадра и его источник.

    Клиент задаёт GSD для исходного кадра; после ужатия пиксель стал крупнее,
    поэтому значение делится на масштаб. Выведенный из камеры GSD уже считается
    по ширине рабочего кадра и пересчёта не требует.
    """
    if options.gsd_m_per_px:
        return options.gsd_m_per_px / scale, "provided"
    if options.camera:
        return (
            georef.gsd_from_camera(
                altitude_m=options.camera.altitude_m,
                focal_mm=options.camera.focal_mm,
                sensor_width_mm=options.camera.sensor_width_mm,
                image_width_px=width_px,
            ),
            "camera",
        )
    return None, "unknown"


def _resolve_georeference(
    options: DetectOptions,
    exif: georef.ExifInfo,
    *,
    width_px: int,
    height_px: int,
    gsd: float | None,
) -> tuple[georef.Georeference | None, str]:
    """Привязка кадра. Углы от клиента > EXIF GPS > центр от клиента > ничего."""
    if options.bounds:
        corners = tuple((float(lat), float(lon)) for lat, lon in options.bounds)
        return (
            georef.Georeference(
                width_px=width_px,
                height_px=height_px,
                corners=corners,  # type: ignore[arg-type]
                approximate=False,
                source="bounds",
            ),
            "bounds",
        )
    if gsd is None:
        return None, "none"
    if exif.lat is not None and exif.lon is not None:
        return (
            georef.georeference_from_center(
                lat=exif.lat,
                lon=exif.lon,
                gsd_m_per_px=gsd,
                width_px=width_px,
                height_px=height_px,
                heading_deg=options.heading_deg,
                source="exif",
            ),
            "exif",
        )
    if options.center is not None:
        lat, lon = options.center
        return (
            georef.georeference_from_center(
                lat=float(lat),
                lon=float(lon),
                gsd_m_per_px=gsd,
                width_px=width_px,
                height_px=height_px,
                heading_deg=options.heading_deg,
                source="center",
            ),
            "center",
        )
    return None, "none"


def _fraud_flags(
    options: DetectOptions, exif: georef.ExifInfo, *, has_detections: bool
) -> list[FraudFlag]:
    """Сигналы для ручной проверки. Ничего не блокируем — только помечаем."""
    flags: list[FraudFlag] = []

    if options.center and exif.lat is not None and exif.lon is not None:
        delta = georef.haversine_m(options.center[0], options.center[1], exif.lat, exif.lon)
        if delta > settings.FRAUD_GPS_DELTA_M:
            flags.append(
                FraudFlag(
                    code="exif_gps_mismatch",
                    severity="warning",
                    message=(
                        f"GPS из EXIF расходится с присланными координатами на {delta:.0f} м"
                    ),
                    details={"delta_m": round(delta, 1), "exif": [exif.lat, exif.lon]},
                )
            )

    captured = _parse_exif_datetime(exif.captured_at)
    if captured is not None:
        age = datetime.now(UTC) - captured
        if age > timedelta(days=settings.FRAUD_MAX_AGE_DAYS):
            flags.append(
                FraudFlag(
                    code="stale_capture_date",
                    severity="warning",
                    message=f"Дата съёмки в EXIF старше {settings.FRAUD_MAX_AGE_DAYS} дней",
                    details={"captured_at": exif.captured_at, "age_days": age.days},
                )
            )
    elif exif.lat is None and exif.lon is None:
        flags.append(
            FraudFlag(
                code="exif_stripped",
                severity="info",
                message="В кадре нет EXIF: ни GPS, ни даты съёмки — проверить происхождение",
            )
        )

    if not has_detections:
        flags.append(
            FraudFlag(
                code="no_detections",
                severity="info",
                message="Модель не нашла мусора на кадре",
            )
        )
    return flags


def _parse_exif_datetime(value: str | None) -> datetime | None:
    """EXIF пишет дату как `2024:07:15 11:03:22`. Часового пояса там нет."""
    if not value:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _build_summary(
    detections: list[Detection], *, label_mask: np.ndarray
) -> Summary:
    """Сводка по кадру: что за мусор, сколько его и какая фракция преобладает."""
    by_class: dict[int, ClassSummary] = {}
    for det in detections:
        row = by_class.get(det.class_id)
        if row is None:
            info = taxonomy.class_info(det.class_id)
            row = ClassSummary(
                class_id=det.class_id,
                label=info.label,
                trash_category=info.trash_category,
                count=0,
                area_px=0,
                area_m2=0.0,
                volume_m3=0.0,
                mass_kg=0.0,
            )
            by_class[det.class_id] = row
        row.count += 1
        row.area_px += det.area_px
        if det.area_m2 is not None:
            row.area_m2 = (row.area_m2 or 0.0) + det.area_m2
            row.volume_m3 = (row.volume_m3 or 0.0) + (det.volume_m3 or 0.0)
            row.mass_kg = (row.mass_kg or 0.0) + (det.mass_kg or 0.0)

    georeferenced = any(d.area_m2 is not None for d in detections)
    if not georeferenced:
        for row in by_class.values():
            row.area_m2 = row.volume_m3 = row.mass_kg = None

    total_area_px = sum(d.area_px for d in detections)
    # Доминирующий класс — по объёму, а если объём неизвестен, по площади:
    # для уборки важнее кубометры, а не число пятен.
    rows = sorted(
        by_class.values(),
        key=lambda r: (r.volume_m3 or 0.0, r.area_px),
        reverse=True,
    )
    dominant = rows[0].trash_category if rows else None

    fractions = [d.fraction for d in detections if d.fraction]
    fraction = None
    if fractions:
        order = ["mega", "macro", "meso", "micro"]
        fraction = min(fractions, key=order.index)

    return Summary(
        count=len(detections),
        by_class=rows,
        total_area_px=total_area_px,
        total_area_m2=sum(d.area_m2 or 0.0 for d in detections) if georeferenced else None,
        total_volume_m3=sum(d.volume_m3 or 0.0 for d in detections) if georeferenced else None,
        total_mass_kg=sum(d.mass_kg or 0.0 for d in detections) if georeferenced else None,
        coverage_ratio=round(total_area_px / float(label_mask.size), 6),
        dominant_category=dominant,
        fraction=fraction,
    )


def _build_candidates(
    detections: list[Detection], options: DetectOptions
) -> list[PointCandidate]:
    """Кандидаты в точки для очереди ООПТ.

    Один кандидат на кадр, а не на пятно: полсотни обрывков сети на десяти
    метрах берега — это одна уборка, а не полсотни задач модератору. Разбивать
    на несколько точек имеет смысл по расстоянию, а этого без привязки не
    сделать, поэтому здесь честно один кадр — одна точка.
    """
    if not detections:
        return []

    summary_categories = sorted({d.trash_category for d in detections})
    geo = [d for d in detections if d.centroid is not None]
    lat = lon = None
    if geo:
        weights = [d.area_px for d in geo]
        total = float(sum(weights)) or 1.0
        lat = round(sum(d.centroid[0] * w for d, w in zip(geo, weights)) / total, 7)
        lon = round(sum(d.centroid[1] * w for d, w in zip(geo, weights)) / total, 7)

    largest = max(detections, key=lambda d: d.volume_m3 or d.area_px)
    volumes = [d.volume_m3 for d in detections if d.volume_m3 is not None]
    masses = [d.mass_kg for d in detections if d.mass_kg is not None]

    return [
        PointCandidate(
            detection_ids=[d.id for d in detections],
            lat=lat,
            lon=lon,
            geometry=largest.geometry,
            trash_categories=summary_categories,
            dominant_category=largest.trash_category,
            fraction=largest.fraction,
            estimated_volume_m3=round(sum(volumes), 4) if volumes else None,
            estimated_mass_kg=round(sum(masses), 2) if masses else None,
            # Уверенность кандидата — по самому крупному объекту, а не средняя:
            # средняя размывается мелочью и занижает очевидные находки.
            confidence=largest.confidence,
            territory_id=options.territory_id,
        )
    ]


def run(
    image: Image.Image,
    options: DetectOptions,
    detector: Detector,
    *,
    job_id: str | None = None,
    georeference: georef.WebMercatorGeoreference | None = None,
) -> PipelineResult:
    """Полный прогон кадра. Никакого ввода-вывода: только вычисления.

    `georeference` передаётся, когда привязка уже известна точно — так делает
    запрос по участку карты: у тайловой сетки координаты выводятся формулой,
    и восстанавливать их из EXIF или углов не нужно.
    """
    started = time.perf_counter()
    timing: dict[str, float] = {}

    original_w, original_h = image.size
    exif_info = georef.read_exif(image)

    work, scale = _fit_to_max_side(image, settings.MAX_SIDE_PX)
    work_w, work_h = work.size

    t0 = time.perf_counter()
    segmentation = detector.segment(work)
    timing["segment"] = round((time.perf_counter() - t0) * 1000, 1)

    gsd_work, gsd_source = _resolve_gsd(options, width_px=work_w, scale=scale)
    if georeference is not None:
        georeference = georeference.rescaled(scale)
        geo_source = "tiles"
    else:
        georeference, geo_source = _resolve_georeference(
            options, exif_info, width_px=work_w, height_px=work_h, gsd=gsd_work
        )
    if gsd_work is None and georeference is not None:
        gsd_work = georeference.gsd_m_per_px()
        gsd_source = "tiles" if geo_source == "tiles" else "bounds"

    t0 = time.perf_counter()
    blobs = masks.extract_blobs(
        segmentation.labels,
        prob_map=segmentation.probs,
        min_area_px=options.min_area_px or settings.MIN_AREA_PX,
    )
    timing["blobs"] = round((time.perf_counter() - t0) * 1000, 1)

    min_conf = (
        options.min_confidence
        if options.min_confidence is not None
        else settings.MIN_CONFIDENCE
    )
    detections: list[Detection] = []
    for index, blob in enumerate(blobs, start=1):
        if blob.confidence < min_conf:
            continue
        info = taxonomy.class_info(blob.class_id)
        m = estimate.measure(
            blob.mask, blob.class_id, gsd_m_per_px=gsd_work, bbox_px=blob.bbox_px
        )
        y0, x0, y1, x1 = blob.bbox_px

        centroid = None
        geometry = None
        if georeference is not None:
            lon, lat = georeference.to_lonlat(*blob.centroid_px)
            centroid = (lat, lon)
            if options.include_geojson:
                geometry = {
                    "type": "Polygon",
                    "coordinates": [georeference.polygon(blob.polygon_px)],
                }

        detections.append(
            Detection(
                id=index,
                class_id=blob.class_id,
                label=info.label,
                label_ru=info.label_ru,
                trash_category=info.trash_category,
                color_hex=info.color_hex,
                confidence=blob.confidence,
                area_px=int(round(m.area_px / scale**2)),
                area_m2=round(m.area_m2, 4) if m.area_m2 is not None else None,
                depth_m=round(m.depth_m, 4) if m.depth_m is not None else None,
                volume_m3=round(m.volume_m3, 4) if m.volume_m3 is not None else None,
                mass_kg=round(m.mass_kg, 2) if m.mass_kg is not None else None,
                fraction=m.fraction,
                centroid_px=(
                    round(blob.centroid_px[0] / scale, 1),
                    round(blob.centroid_px[1] / scale, 1),
                ),
                bbox_px=(
                    int(x0 / scale),
                    int(y0 / scale),
                    int(x1 / scale),
                    int(y1 / scale),
                ),
                polygon_px=[(round(x / scale, 1), round(y / scale, 1)) for x, y in blob.polygon_px],
                centroid=centroid,
                geometry=geometry,
            )
        )

    summary = _build_summary(detections, label_mask=segmentation.labels)
    # Кандидаты — это работа для сотрудника ООПТ. Если снимок заведомо слишком
    # груб для модели, находки остаются в ответе (посмотреть глазами полезно),
    # но в очередь не идут: мусорные задачи в очереди хуже, чем их отсутствие.
    candidates = [] if options.suppress_candidates else _build_candidates(detections, options)
    flags = _fraud_flags(options, exif_info, has_detections=bool(detections))

    geojson = None
    if options.include_geojson and georeference is not None:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": d.geometry,
                    "properties": {
                        "id": d.id,
                        "class": d.label,
                        "label_ru": d.label_ru,
                        "trash_category": d.trash_category,
                        "confidence": d.confidence,
                        "area_m2": d.area_m2,
                        "volume_m3": d.volume_m3,
                        "mass_kg": d.mass_kg,
                        "fraction": d.fraction,
                        "color": d.color_hex,
                    },
                }
                for d in detections
                if d.geometry is not None
            ],
        }

    overlay_png = blend_png = None
    if options.render_overlay:
        t0 = time.perf_counter()
        overlay = masks.render_overlay(
            segmentation.labels, by_class=options.overlay_by_class
        )
        if scale != 1.0:
            overlay = overlay.resize((original_w, original_h), Image.Resampling.NEAREST)
        overlay_png = _to_png(overlay)
        blend_png = _to_png(masks.render_blend(work, segmentation.labels))
        timing["overlay"] = round((time.perf_counter() - t0) * 1000, 1)

    backend_info = detector.info()
    timing["total"] = round((time.perf_counter() - started) * 1000, 1)

    return PipelineResult(
        job_id=job_id or uuid.uuid4().hex,
        model=ModelInfo(
            backend=backend_info.name,
            version=backend_info.version,
            weights=backend_info.weights,
            device=backend_info.device,
            trained=backend_info.trained,
            notes=backend_info.notes,
        ),
        image=ImageInfo(
            width=original_w,
            height=original_h,
            processed_width=work_w,
            processed_height=work_h,
            # Наружу — GSD исходного кадра: клиент про ужатие знать не обязан.
            gsd_m_per_px=round(gsd_work * scale, 6) if gsd_work else None,
            gsd_source=gsd_source,  # type: ignore[arg-type]
            georeferenced=georeference is not None,
            georeference_source=geo_source,  # type: ignore[arg-type]
            georeference_approximate=bool(georeference and georeference.approximate),
        ),
        detections=detections,
        summary=summary,
        point_candidates=candidates,
        fraud_flags=flags,
        exif=ExifOut(**exif_info.__dict__),
        geojson=geojson,
        overlay_png=overlay_png,
        blend_png=blend_png,
        overlay_bounds=georeference.bounds_lonlat() if georeference else None,
        timing_ms=timing,
        label_mask=segmentation.labels,
    )


def _to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
