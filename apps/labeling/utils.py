from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from django.conf import settings

logger = logging.getLogger(__name__)


def resolve_labeling_database_path() -> str:
    """Resolve the database path used for labeling operations."""
    labeling_db: Optional[str] = getattr(settings, "LABELING_DB_PATH", None)
    base_dir = Path(getattr(settings, "BASE_DIR", "."))

    if labeling_db:
        candidate = Path(labeling_db)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.exists():
            logger.debug("Using configured LABELING_DB_PATH at %s", candidate)
            return str(candidate)
        logger.warning(
            "Configured LABELING_DB_PATH %s does not exist; falling back to defaults",
            candidate,
        )

    default_name = Path(str(settings.DATABASES["default"]["NAME"]))
    if not default_name.is_absolute():
        default_name = base_dir / default_name

    fallback_candidates = [
        base_dir / "total.db",
        base_dir / "core" / "total.db",
        default_name,
        base_dir / "messages.db",
    ]

    for candidate in fallback_candidates:
        if candidate.exists():
            logger.debug("Resolved labeling database to %s", candidate)
            return str(candidate)

    logger.warning(
        "No labeling database candidates found; defaulting to %s", default_name
    )
    return str(default_name)
