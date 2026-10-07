# RoadLens

## Docker Compose

Start Docker Desktop with Linux containers enabled. From the repository root:

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs
- Backend health: http://localhost:8000/health
- Proxied health: http://localhost:3000/api/health

Both health URLs should return `{"status":"healthy"}`. Compose creates a shared
network and waits for backend health before starting the frontend. Browser code
can use `fetch("/api/health")`; Next.js forwards `/api/...` to `http://backend:8000/...`.
These same-origin browser requests need no CORS configuration.

`BACKEND_URL` is a frontend build argument because rewrites are generated at build
time. Rebuild after changing it. Local Next.js development defaults to
`http://localhost:8000`.

Stop with Ctrl+C, then run `docker compose down` to remove containers and the network.
Use `docker compose up --build -d` for background operation and `docker compose logs -f`
for logs. Ports 3000 and 8000 must be free. Builds need network access for dependencies
and the existing Google fonts.
