"use strict";

(async function bootstrapSokomind() {
  const response = await fetch("build.json", {cache: "no-store"});
  if (!response.ok) throw new Error(`Build manifest request failed (${response.status}).`);
  const manifest = await response.json();
  if (!/^\d{4}-\d{2}-\d{2}\.\d+$/.test(manifest.build)) {
    throw new Error("Build manifest contains an invalid revision.");
  }
  globalThis.SOKOMIND_BUILD = manifest.build;
  const revision = `?build=${encodeURIComponent(manifest.build)}`;

  const stylesheet = document.createElement("link");
  stylesheet.rel = "stylesheet";
  stylesheet.href = `styles.css${revision}`;
  document.head.append(stylesheet);

  const scripts = [
    "levels.js",
    "game-state.js",
    "path-validation.js",
    "keyboard-policy.js",
    "search-log.js",
    "director-policy.js",
    "solver-director.js",
    "app.js",
  ];
  for (const source of scripts) {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `${source}${revision}`;
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Could not load ${source}.`));
      document.body.append(script);
    });
  }
})().catch(error => {
  const status = document.getElementById("status");
  if (status) status.textContent = `Sokomind failed to start: ${error.message}`;
  console.error(error);
});
