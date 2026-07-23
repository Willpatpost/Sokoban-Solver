"use strict";

const {defineConfig, devices} = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests/browser",
  timeout: 30000,
  expect: {timeout: 5000},
  fullyParallel: false,
  reporter: process.env.CI ? "line" : "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
  },
  globalSetup: require.resolve("./tests/browser/global-setup.js"),
  projects: [
    {name: "chromium", use: {...devices["Desktop Chrome"]}},
    {name: "webkit", use: {...devices["Desktop Safari"]}},
  ],
});
