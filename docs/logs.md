# RoadLens Setup Log

## 2026-10-08

### PostgreSQL foundation

- Selected PostgreSQL and documented the upload-first data plan in `docs/data-plan.md`.
- Added PostgreSQL 17.11 to Compose with a named volume, loopback-only port 5432 and a readiness check.
- Added `.env.example` and local ignored `.env` credentials; excluded environment files from Docker build context.
- Added psycopg connectivity and `/health/db`, returning HTTP 503 without connection details on failure. Existing `/health` remains a process check.
- Configured the backend to wait for PostgreSQL and the frontend to wait for database-aware backend readiness.
- Started the database successfully and verified PostgreSQL 17.11 and a real authenticated backend query.
- Five focused tests and Ruff checks passed. Compose configuration validated. Full frontend/backend container rebuild was not part of this verification.
- Application tables, migrations and PostGIS are not added yet; the next milestone is the initial upload schema.

## 2026-10-06

### Repository and tooling

- Created the repository structure for the backend, frontend, machine-learning code, shared code, notebooks, tests, and documentation.
- Added Python version pinning for Python 3.11 through `.python-version`.
- Added `pyproject.toml` with project metadata and the current Python dependency set.
- Added `uv.lock` to lock Python dependencies.
- Added a local `.venv` Python environment.
- Added development dependencies for `pytest` and `ruff`.
- Added `.dockerignore`.

### Backend

- Added a FastAPI application in `backend/app/main.py`.
- Configured the API title as `RoadLens API` and version `0.1.0`.
- Added `GET /health`, returning `{ "status": "healthy" }`.
- Created initial package directories for models, routes, schemas, and services.
- Added `backend/Dockerfile` using `python:3.11-slim`.
- Configured the Docker image to install dependencies with `uv`, expose port `8000`, and run the API with Uvicorn.

### Frontend

- Added a Next.js application under `frontend/web`.
- Configured Next.js `16.3.8`, React `19.2.8`, TypeScript, Tailwind CSS 4, ESLint, and the React compiler plugin.
- Added the standard frontend scripts: `dev`, `build`, `start`, and `lint`.
- Added the initial App Router files: `layout.tsx`, `page.tsx`, and `globals.css`.
- Added the default Next.js public assets and frontend `package-lock.json`.
- The current page is still the generated Next.js starter page; RoadLens UI functionality has not been implemented yet.

### Documentation

- Added placeholder files for architecture documentation and interface documentation.
- Added this setup log to track completed project initialization work.

### Current status

- Project initialization is complete for the backend and frontend scaffolds.
- Machine-learning modules, shared application logic, domain models, API routes beyond `/health`, tests, and production UI are not implemented yet.

## 2026-10-07

### Frontend Docker

- Added a multi-stage Node.js 22 Dockerfile with npm ci and a non-root runtime on port 3000.
- Enabled Next.js standalone output, including public and static assets in the image.
- Added frontend .dockerignore and build/run instructions in frontend/web/README.md.
- Production build verification was blocked by Google Fonts connectivity. Docker build/run verification remains pending because the Docker engine was unavailable.

### Compose integration

- Added root compose.yaml to build and start backend and frontend on a shared network.
- Added backend readiness check and frontend startup dependency.
- Added /api path forwarding in Next.js, using a build-time BACKEND_URL supplied by Compose.
- Documented startup and health-check URLs in the root README.
- Frontend configuration lint and diff whitespace checks passed; runtime verification is limited by Docker access in the agent environment.
