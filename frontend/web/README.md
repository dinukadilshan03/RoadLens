This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Docker

Start Docker Desktop with Linux containers enabled. From the repository root:

```bash
docker build -t roadlens-frontend ./frontend/web
docker run --rm --name roadlens-frontend -p 3000:3000 roadlens-frontend
```

Open http://localhost:3000. Stop the foreground container with Ctrl+C.
If port 3000 is occupied, use `-p 3001:3000` and open http://localhost:3001.

The build context must be `frontend/web`. The multi-stage image installs locked
dependencies with `npm ci`, builds Next.js, and copies its standalone server,
static assets, and public files into a Node.js 22 runtime running as a non-root user.
Local dependencies, build output, and environment files are excluded from the context.

This is a production build; rebuild the image after source changes. For hot reload,
continue using `npm run dev` locally. The build needs network access for npm packages
and the Google fonts used by the current layout. Run both services with `docker compose up --build` from the repository root.
Browser requests to `/api/...` are forwarded to the backend (see the root README).
For standalone Docker builds, pass `--build-arg BACKEND_URL=http://BACKEND_HOST:8000`
with a backend hostname reachable from the frontend container. This URL is baked
into the build; it defaults to localhost:8000 for local development.
