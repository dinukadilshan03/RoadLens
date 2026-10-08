# RoadLens

## Docker Compose

PostgreSQL is configured as the `db` service. Before the first start, copy
`.env.example` to `.env` and replace `POSTGRES_PASSWORD` with a local password.
If `.env` already exists, keep it. It is ignored by Git and excluded from Docker builds.

Start Docker Desktop with Linux containers enabled. From the repository root:

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs
- Backend health: http://localhost:8000/health
- Database readiness: http://localhost:8000/health/db
- Proxied health: http://localhost:3000/api/health

Both health URLs should return `{"status":"healthy"}`. Compose creates a shared
network and waits for backend health before starting the frontend. Browser code
can use `fetch("/api/health")`; Next.js forwards `/api/...` to `http://backend:8000/...`.
These same-origin browser requests need no CORS configuration.

The backend waits for PostgreSQL, and the frontend waits for backend database
readiness. `/health` checks only the API process; `/health/db` runs a database
query and returns `{"status":"healthy","database":"connected"}`, or HTTP 503
when the database is unavailable.

## Database-only development

Start only PostgreSQL (no frontend build required):

```powershell
docker compose up -d --wait db
docker compose exec db psql -U roadlens -d roadlens -c "SELECT current_database(), version();"
```

The default database and user are both `roadlens`. Substitute your values if
you changed them in `.env`. PostgreSQL is available at `127.0.0.1:5432` for
local database clients; the backend container uses `db:5432`.

Run the backend locally in another terminal:

```powershell
uv sync --frozen
uv run uvicorn backend.app.main:app --reload
```

The backend loads root `.env` automatically. Existing environment variables
take precedence. Open http://localhost:8000/health/db to verify the connection.
Alternatively, use `docker compose up -d --build backend` to run it in Docker.

The named `postgres_data` volume keeps the database across container restarts
and `docker compose down`. **Do not use `docker compose down -v` unless you
intend to delete database data.** A volume is not a backup.

Credentials initialize an empty volume only. Editing `.env` later does not
change an existing database password; change the database role password too.
This configuration uses the image's bootstrap user for the local demo; use a
separate restricted application role before production deployment.

This step sets up an empty database and connectivity only. Application tables
and migrations will follow the upload-first [data plan](docs/data-plan.md).

## Build notes

`BACKEND_URL` is a frontend build argument because rewrites are generated at build
time. Rebuild after changing it. Local Next.js development defaults to
`http://localhost:8000`.

Stop with Ctrl+C, then run `docker compose down` to remove containers and the network.
Use `docker compose up --build -d` for background operation and `docker compose logs -f`
for logs. Ports 3000 and 8000 must be free. Builds need network access for dependencies
and the existing Google fonts.
