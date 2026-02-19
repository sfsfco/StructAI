"""
Celery worker entrypoint.

Usage
-----
Default worker (all queues)::

    celery -A worker.worker worker --loglevel=info -Q default,indexing,maintenance

Indexing-only worker (scale separately)::

    celery -A worker.worker worker --loglevel=info -Q indexing -c 2

Beat scheduler (run once alongside workers)::

    celery -A worker.worker beat --loglevel=info

Combined worker + beat (dev convenience)::

    celery -A worker.worker worker --beat --loglevel=info -Q default,indexing,maintenance
"""

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.workers.tasks import celery_app  # noqa: F401 — Celery discovers tasks via this import

settings = get_settings()

# Ensure structured logging is initialised even before the first task runs.
setup_logging(log_level=settings.LOG_LEVEL, json_logs=True)
