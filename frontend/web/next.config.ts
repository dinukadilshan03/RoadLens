import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactCompiler: true,
  // Production rewrites are generated at build time.
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
    return [
      // Versioned domain API: forwarded unchanged (/api/v1/... -> backend /api/v1/...).
      { source: "/api/v1/:path*", destination: `${backendUrl}/api/v1/:path*` },
      // Existing behaviour for other paths, e.g. /api/health -> backend /health.
      { source: "/api/:path*", destination: `${backendUrl}/:path*` },
    ];
  },
};

export default nextConfig;
