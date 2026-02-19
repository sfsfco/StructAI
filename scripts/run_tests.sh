#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  run_tests.sh — Convenience script for running the StructAI test suite
# ═══════════════════════════════════════════════════════════════════════
#
#  Usage:
#    ./scripts/run_tests.sh              # Run all tests
#    ./scripts/run_tests.sh unit         # Unit tests only (no deps needed)
#    ./scripts/run_tests.sh integration  # Integration tests (needs PG + Redis)
#    ./scripts/run_tests.sh e2e          # End-to-end tests (needs PG + Redis)
#    ./scripts/run_tests.sh coverage     # Full suite with coverage report
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Test environment variables
export OPENAI_API_KEY="${OPENAI_API_KEY:-test-key-not-real}"
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export REDIS_HOST="${REDIS_HOST:-localhost}"
export DEBUG=true
export LOG_LEVEL=WARNING

# Integration / E2E test database (separate from dev)
export TEST_DATABASE_URL="${TEST_DATABASE_URL:-postgresql+asyncpg://ai_user:ai_pass@localhost:5433/ai_db_test}"
export TEST_REDIS_URL="${TEST_REDIS_URL:-redis://localhost:6380/1}"

MODE="${1:-all}"

case "$MODE" in
    unit)
        echo "🧪 Running unit tests..."
        python -m pytest tests/unit -m unit -v --tb=short
        ;;
    integration)
        echo "🧪 Running integration tests..."
        echo "   (Ensure test DB and Redis are running: docker compose -f docker-compose.test.yml up -d)"
        python -m pytest tests/integration -m integration -v --tb=short
        ;;
    e2e)
        echo "🧪 Running E2E tests..."
        echo "   (Ensure test DB and Redis are running: docker compose -f docker-compose.test.yml up -d)"
        python -m pytest tests/e2e -m e2e -v --tb=short
        ;;
    coverage)
        echo "🧪 Running full test suite with coverage..."
        python -m pytest tests/ -v --tb=short \
            --cov=app \
            --cov-report=term-missing \
            --cov-report=html:htmlcov \
            --cov-fail-under=70
        echo "📊 Coverage report: htmlcov/index.html"
        ;;
    all)
        echo "🧪 Running all tests..."
        python -m pytest tests/ -v --tb=short
        ;;
    *)
        echo "Usage: $0 {unit|integration|e2e|coverage|all}"
        exit 1
        ;;
esac
