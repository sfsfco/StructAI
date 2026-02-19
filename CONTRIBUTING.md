# Contributing to StructAI

Thank you for your interest in contributing to StructAI! This guide covers everything you need to set up your development environment, follow our coding standards, and submit changes.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Project Layout](#project-layout)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Architecture Decisions](#architecture-decisions)

---

## Development Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose v2+
- An OpenAI API key

### 1. Clone & Install

```bash
git clone https://github.com/your-org/StructAI.git
cd StructAI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Development extras
pip install pytest pytest-asyncio pytest-cov httpx black ruff mypy
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 3. Start Dependencies

```bash
# Start PostgreSQL + Redis for development
docker compose up -d db redis
```

### 4. Run the API Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run Workers Locally

```bash
celery -A worker.worker worker --loglevel=info -Q indexing,default,maintenance
```

---

## Project Layout

```
app/
├── api/           # HTTP layer — routes, request/response handling
├── core/          # Framework config, logging, metrics
├── db/            # Database models, repository pattern, session management
├── middleware/     # Request middleware (CORS, metrics, rate limiting, backpressure)
├── schemas/       # Pydantic schemas for validation
├── services/      # Business logic layer — LLM, embeddings, cache, vector store
└── workers/       # Celery task definitions
```

**Key design principle:** The `services/` layer contains all business logic. The `api/` layer is thin — it validates input, calls services, and returns responses. The `db/` layer handles persistence only.

---

## Coding Standards

### Python Style

- **Formatter:** [Black](https://black.readthedocs.io/) with default settings (line length 88)
- **Linter:** [Ruff](https://docs.astral.sh/ruff/) for fast linting
- **Type checker:** [mypy](https://mypy-lang.org/) in strict mode for new code

```bash
# Format
black app/ tests/ worker/

# Lint
ruff check app/ tests/ worker/

# Type check
mypy app/
```

### Conventions

| Rule | Example |
|------|---------|
| Async everywhere | Use `async def` for all service methods and routes |
| Type annotations | All function signatures must have type hints |
| Docstrings | Module-level + all public classes/functions (Google style) |
| Imports | stdlib → third-party → local, separated by blank lines |
| Constants | UPPER_SNAKE_CASE |
| Classes | PascalCase |
| Functions/variables | snake_case |

### Logging

Use `structlog` bound loggers — never `print()` or stdlib `logging` directly:

```python
import structlog
logger = structlog.get_logger(__name__)

async def my_function():
    logger.info("operation.started", document_id=doc_id, chunk_count=len(chunks))
```

### Error Handling

- Raise `HTTPException` in the API layer only
- Services raise domain-specific exceptions
- Background tasks log failures and update document status
- Never swallow exceptions silently

---

## Testing

### Test Categories

| Type | Location | Dependencies | Speed |
|------|----------|-------------|-------|
| Unit | `tests/unit/` | None (all mocked) | < 10s |
| Integration | `tests/integration/` | PostgreSQL + Redis | < 60s |
| E2E | `tests/e2e/` | Full stack | < 120s |

### Running Tests

```bash
# Unit tests (fast, no dependencies)
pytest tests/unit/ -v

# Integration tests (needs test DB + Redis)
docker compose -f docker-compose.test.yml up -d
TEST_DATABASE_URL=postgresql+asyncpg://ai_user:ai_pass@localhost:5433/ai_db_test \
TEST_REDIS_URL=redis://localhost:6380/1 \
pytest tests/integration/ -v
docker compose -f docker-compose.test.yml down -v

# All tests with coverage
pytest --cov=app --cov-report=term-missing
```

### Writing Tests

- **Unit tests:** Mock all external dependencies (DB, Redis, OpenAI, FAISS)
- **Integration tests:** Use real DB + Redis from `docker-compose.test.yml`
- **Use fixtures:** Leverage `conftest.py` fixtures for common setup
- **Async tests:** Mark with `@pytest.mark.asyncio`
- **Naming:** `test_<what>_<condition>_<expected>` (e.g., `test_extract_missing_document_returns_404`)

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

### Types

| Type | When |
|------|------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring (no behaviour change) |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `chore` | Build, CI, deps, tooling |
| `perf` | Performance improvement |

### Examples

```
feat(api): add readiness probe with dependency checks
fix(worker): prevent duplicate chunk indexing on retry
refactor(services): extract BaseVectorStore interface
docs: add Kubernetes deployment guide
test(unit): add LLM client retry tests
chore: upgrade FastAPI to 0.115.6
```

---

## Pull Request Process

1. **Branch from `main`:** `git checkout -b feat/my-feature`
2. **Keep PRs focused:** One feature or fix per PR
3. **Write tests:** All new code should have corresponding tests
4. **Update docs:** If you change the API, update the README
5. **Run the full suite:**
   ```bash
   black app/ tests/ worker/ --check
   ruff check app/ tests/ worker/
   pytest --cov=app
   ```
6. **PR description:** Explain what, why, and any trade-offs
7. **Request review:** Tag a maintainer

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code formatted (`black --check`)
- [ ] Linting clean (`ruff check`)
- [ ] New env vars added to `.env.example`
- [ ] README updated if API changed
- [ ] No secrets or API keys committed

---

## Architecture Decisions

When making significant architectural changes, document the decision:

1. **What** problem are you solving?
2. **Why** this approach over alternatives?
3. **Trade-offs** — what does this decision cost?

Key patterns used in this project:

- **Repository pattern** — Data access abstracted behind `DocumentRepository`
- **Dependency injection** — `FastAPI Depends()` for all service instantiation
- **Abstract interfaces** — `BaseLLMClient`, `BaseVectorStore` for provider swapping
- **Celery chains** — Pipeline stages as independent, retryable tasks
- **Three-tier caching** — Extraction → Embedding → Dedup with different TTLs

---

## Questions?

Open an issue with the `question` label or start a discussion in the repository.
