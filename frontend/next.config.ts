import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone",
  experimental: { cpus: 2 },
  async rewrites() {
    const origin = process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8000";
    return [{ source: "/api/v1/:path*", destination: `${origin}/api/v1/:path*` }];
  },
};

export default config;
