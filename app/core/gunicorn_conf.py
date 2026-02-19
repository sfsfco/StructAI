"""
Gunicorn configuration for production multi-worker deployment.

Usage::

    gunicorn app.main:app -c app/core/gunicorn_conf.py

Environment variables
---------------------
  WEB_CONCURRENCY  — number of worker processes (default: CPU count × 2 + 1)
  BIND             — host:port to bind (default: 0.0.0.0:8000)
  GRACEFUL_TIMEOUT — seconds to wait for workers to finish (default: 30)
  KEEP_ALIVE       — keep-alive timeout (default: 5)
  LOG_LEVEL        — gunicorn log level (default: info)

Notes
-----
  - Uses ``uvicorn.workers.UvicornWorker`` for ASGI compatibility.
  - ``preload_app = True`` shares the loaded model/index across forks,
    reducing per-worker memory.
  - ``prometheus_multiproc_dir`` must be set when using prometheus_client
    in a pre-fork server (gunicorn calls ``child_exit`` to clean up).
  - ``max_requests`` + jitter triggers graceful worker recycling to prevent
    memory leaks from accumulating over long-running processes.
"""

from __future__ import annotations

import multiprocessing
import os

# ── Bind & Workers ───────────────────────────────────────────────────────

bind = os.getenv("BIND", "0.0.0.0:8000")

# Rule of thumb: 2× CPU + 1 for I/O-bound workloads.
# For GPU/LLM-heavy workloads, use fewer workers (e.g. CPU count).
_cpu_count = multiprocessing.cpu_count()
workers = int(os.getenv("WEB_CONCURRENCY", _cpu_count * 2 + 1))

worker_class = "uvicorn.workers.UvicornWorker"

# ── Timeouts ─────────────────────────────────────────────────────────────

# Graceful timeout: how long to wait for in-flight requests on shutdown.
graceful_timeout = int(os.getenv("GRACEFUL_TIMEOUT", "30"))

# Keep-alive allows connection reuse from reverse proxies / load balancers.
keepalive = int(os.getenv("KEEP_ALIVE", "5"))

# Request timeout — kill workers that take too long (LLM calls can be slow).
timeout = int(os.getenv("WORKER_TIMEOUT", "120"))

# ── Worker Lifecycle ─────────────────────────────────────────────────────

# Recycle workers after N requests to prevent memory leaks.
max_requests = int(os.getenv("MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "50"))

# Pre-load the application before forking workers.
# Saves memory (copy-on-write) and catches startup errors early.
preload_app = True

# ── Logging ──────────────────────────────────────────────────────────────

loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"   # stdout
errorlog = "-"    # stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# ── Prometheus Multi-Process Mode ────────────────────────────────────────

# prometheus_client requires a shared directory for aggregating metrics
# across gunicorn workers.  Set this env var before starting gunicorn.
_prom_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc")
os.makedirs(_prom_dir, exist_ok=True)
os.environ["PROMETHEUS_MULTIPROC_DIR"] = _prom_dir


def child_exit(server, worker):  # noqa: ARG001
    """
    Called when a worker process exits.

    Cleans up the per-worker prometheus metrics files so stale data
    does not pollute aggregated metrics.
    """
    try:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(worker.pid)
    except Exception:
        pass


# ── Server Hooks ─────────────────────────────────────────────────────────


def on_starting(server):
    """Called just before the master process is initialised."""
    server.log.info("Gunicorn master starting (workers=%d)", workers)


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Gunicorn ready — listening on %s", bind)


def post_fork(server, worker):
    """Called in each worker process after fork."""
    server.log.info("Worker spawned (pid=%d)", worker.pid)
