# RoadLens frontend

This directory contains the initial Next.js application shell for RoadLens. It is set up as a UI scaffold and proxy layer for the backend, but it does not yet implement the RoadLens product experience.

## Current status

- Next.js app scaffold exists under `frontend/web`.
- Docker and Compose wiring forwards browser requests for `/api/...` to the backend service.
- The page still reflects the default create-next-app starter content rather than RoadLens-specific screens.
- The actual inspection workflow, data upload UI, map integration, and result views are not implemented yet.

## Local development

From this directory:

```bash
npm install
npm run dev
```

Then open http://localhost:3000.

## Docker

From the repository root:

```bash
docker compose up --build
```

This starts the frontend and backend together. The frontend is exposed at http://localhost:3000 and proxies API requests to the backend on port 8000.

## Build notes

- The frontend image uses a multi-stage Node.js 22 build and runs as a non-root user.
- It expects a `BACKEND_URL` build argument for production container builds.
- The project still depends on upstream Next.js defaults and the generated starter screen until product UI work begins.
