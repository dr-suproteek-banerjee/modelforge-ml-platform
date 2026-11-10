import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  timeout: 90_000,
  use: {
    ...devices["Desktop Chrome"],
    baseURL: process.env.MF_WEB_URL ?? "http://127.0.0.1:3000",
    launchOptions: process.env.MF_CHROMIUM ? { executablePath: process.env.MF_CHROMIUM } : {},
  },
});
