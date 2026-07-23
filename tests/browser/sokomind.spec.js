"use strict";

const {test, expect} = require("@playwright/test");
const {build: EXPECTED_BUILD} = require("../../docs/build.json");

async function enterGame(page) {
  await page.goto("/");
  await expect(page.locator("#solver-build")).toHaveText(EXPECTED_BUILD);
  await page.getByRole("button", {name: /Play Sokomind/}).click();
  await expect(page.locator("#home-screen")).toHaveClass(/hidden/);
}

function installScriptedWorker(page, mode) {
  return page.addInitScript(selectedMode => {
    window.__workerMessages = [];
    if (selectedMode === "silent") window.SOKOMIND_WORKER_WATCHDOG_MS = 80;
    let bridgeNumber = 0;
    let milestoneWorker = null;
    let milestoneState = null;
    const emitLater = (worker, data, delay = 0, ignoreTermination = false) => {
      setTimeout(() => {
        if (!worker.terminated || ignoreTermination) worker.onmessage?.({data});
      }, delay);
    };
    class ScriptedWorker {
      constructor() {
        this.terminated = false;
        this.onmessage = null;
        this.onerror = null;
      }
      terminate() { this.terminated = true; }
      postMessage(payload) {
        window.__workerMessages.push(structuredClone(payload));
        if (selectedMode === "stale") {
          emitLater(this, {
            type: "done", path: ["Down"], visited: 1,
            status: "solved", terminationReason: "solution",
          }, 500, true);
          return;
        }
        if (selectedMode === "error") {
          setTimeout(() => this.onerror?.(new ErrorEvent("error", {
            message: "scripted worker failure",
          })), 0);
          return;
        }
        if (selectedMode === "silent") return;
        if (payload.algorithm === "analyze-puzzle") {
          emitLater(this, {
            type: "done",
            analysis: {
              difficulty: "complex",
              dimensions: {columns: 5, rows: 5},
              boxes: 1,
              initialHeuristic: 1,
              legalPushes: 1,
              rooms: [],
              articulations: 1,
              tunnelCells: 1,
              surplusBoxes: 0,
              pressure: 1,
              reverseStartRegions: 1,
              productiveReverseStartRegions: 1,
              reverseStartPulls: 1,
              phases: [{id: "evacuation", reason: "scripted browser fixture"}],
              searchScale: 1,
              recommendations: {
                reverseWorkerLimit: 1,
                sideVisitedLimit: 20,
                beamAttempts: 0,
                beamWidth: 20,
                beamVisited: 20,
                useEvacuation: true,
                useSequenceMacros: true,
                useMilestoneReverse: true,
                checkpointLimit: 2,
              },
            },
          });
          return;
        }
        if (selectedMode === "priority" && payload.handoffStage === "structural") {
          emitLater(this, {
            type: "done", path: null, cutoff: true, visited: 1, checkpoints: [],
            status: "cutoff", terminationReason: "fixture-structural",
          }, 120);
          return;
        }
        if (selectedMode === "anytime" && payload.handoffStage === "structural") {
          emitLater(this, {
            type: "done",
            path: ["Left", "Down", "Up", "Right", "Down"],
            visited: 1, checkpoints: [],
            status: "solved", terminationReason: "solution",
          });
          return;
        }
        if (selectedMode === "anytime" &&
            payload.algorithm === "solution-window-rewrite") {
          emitLater(this, {
            type: "done", path: ["Down"], visited: 1,
            initialPushes: 1, initialMoves: 1, bestPushes: 1, bestMoves: 1,
            improvements: 0, status: "solved",
            terminationReason: "rewrite-fixed-point",
          }, 10);
          return;
        }
        const checkpoint = {
          state: payload.state,
          path: [],
          cost: 0,
          estimate: 1,
        };
        if (payload.targetedReverse) {
          milestoneWorker = this;
          milestoneState = payload.state;
          emitLater(this, {
            type: "landmarks",
            landmarks: [{id: "compatible-layout", state: payload.state, cost: 0, estimate: 1}],
          });
          return;
        }
        if (payload.handoffStage === "evacuation") {
          emitLater(this, {
            type: "done", path: null, cutoff: true, visited: 1,
            phaseCheckpoint: checkpoint, checkpoints: [],
            status: "cutoff", terminationReason: "fixture-checkpoint",
          });
          return;
        }
        if (payload.handoffStage === "packing") {
          emitLater(this, {
            type: "done", path: null, cutoff: true, visited: 1, checkpoints: [],
            status: "cutoff", terminationReason: "fixture-packing",
          }, 5);
          return;
        }
        if (payload.handoffStage === "bridge") {
          bridgeNumber++;
          emitLater(this, {
            type: "done", path: null, cutoff: true, visited: 2,
            status: "cutoff",
            terminationReason: bridgeNumber === 1 ? "bridge-budget" : "target-incompatible",
          });
          if (bridgeNumber === 1) {
            emitLater(milestoneWorker, {
              type: "landmarks",
              landmarks: [{
                id: "incompatible-layout",
                state: milestoneState,
                cost: 0,
                estimate: 2,
              }],
            }, 1);
            emitLater(milestoneWorker, {
              type: "done", path: null, cutoff: true, visited: 1,
              status: "cutoff", terminationReason: "fixture-complete",
            }, 20);
          }
          return;
        }
        emitLater(this, {
          type: "done", path: null, cutoff: true, visited: 1,
          status: "cutoff", terminationReason: "fixture-side-complete",
        });
      }
    }
    Object.defineProperty(navigator, "hardwareConcurrency", {value: 4, configurable: true});
    if (selectedMode === "priority") {
      Object.defineProperty(navigator, "deviceMemory", {value: 8, configurable: true});
    }
    window.Worker = ScriptedWorker;
  }, mode);
}

test("page load, selection, keyboard completion, stored bounds, and build display", async ({page}) => {
  await enterGame(page);
  await expect(page.locator("#solver-build")).toHaveText(EXPECTED_BUILD);
  await page.getByRole("button", {name: /Tiny/}).nth(1).click();
  await expect(page.locator("#level-title")).toHaveText("Tiny");
  await page.getByRole("button", {name: /Ultra Tiny/}).click();
  await page.locator("#board").press("ArrowDown");
  await expect(page.locator("#complete-dialog")).toBeVisible();
  await expect(page.locator("#status")).toHaveText("Solved in 1 moves!");
  await expect.poll(() => page.evaluate(() =>
    JSON.parse(localStorage.getItem("sokomind-push-bounds-v1"))["ultra-tiny"])).toBe(1);
});

test("actual worker supports hint, solve, stop, undo, and reset during animation", async ({page}) => {
  await enterGame(page);
  await page.locator("#algorithm").selectOption("push-astar");
  await page.getByRole("button", {name: "Hint"}).click();
  await expect(page.locator("#status")).toContainText("Hint: Down");

  await page.getByRole("button", {name: /Tiny/}).nth(1).click();
  await page.getByRole("button", {name: "Solve"}).click();
  await expect(page.locator("#status")).toContainText("Found");
  await page.getByRole("button", {name: "Undo"}).click();
  await expect(page.locator("#status")).toHaveText("Undid one move.");

  await page.getByRole("button", {name: "Solve"}).click();
  await expect(page.locator("#status")).toContainText("Found");
  await page.getByRole("button", {name: "Reset"}).click();
  await expect(page.locator("#status")).toHaveText("Level reset.");
  await expect(page.locator("#move-count")).toHaveText("0");

  await page.getByRole("button", {name: /Huge/}).click();
  await page.getByRole("button", {name: "Solve"}).click();
  await page.getByRole("button", {name: "Stop"}).click();
  await expect(page.locator("#status")).toHaveText("Stopped.");
});

test("touch controls and responsive board sizing work at the mobile breakpoint", async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await enterGame(page);
  await expect(page.locator(".touch-controls")).toHaveCSS("display", "grid");
  await page.getByRole("button", {name: "Move down"}).dispatchEvent("pointerdown");
  await expect(page.locator("#complete-dialog")).toBeVisible();
  await expect(page.locator("#board")).toHaveCSS("--tile-size", /px/);
});

test("stale worker messages cannot mutate a stopped search", async ({page}) => {
  await installScriptedWorker(page, "stale");
  await enterGame(page);
  await page.locator("#algorithm").selectOption("push-astar");
  await page.getByRole("button", {name: "Solve"}).click();
  await page.getByRole("button", {name: "Stop"}).click();
  await page.waitForTimeout(650);
  await expect(page.locator("#status")).toHaveText("Stopped.");
  await expect(page.locator("#move-count")).toHaveText("0");
});

test("worker errors surface as an explicit failed search", async ({page}) => {
  await installScriptedWorker(page, "error");
  await enterGame(page);
  await page.locator("#algorithm").selectOption("push-astar");
  await page.getByRole("button", {name: "Solve"}).click();
  await expect(page.locator("#status")).toHaveText("Solver worker failed.");
  await expect(page.locator("#search-log-text")).toHaveValue(/worker failed/);
});

test("silent standard workers are retired by the liveness watchdog", async ({page}) => {
  await installScriptedWorker(page, "silent");
  await enterGame(page);
  await page.locator("#algorithm").selectOption("push-astar");
  await page.getByRole("button", {name: "Solve"}).click();
  await expect(page.locator("#status")).toHaveText("Solver worker stopped responding.");
  await expect(page.locator("#search-log-text")).toHaveValue(/worker-watchdog/);
  await expect(page.getByRole("button", {name: "Solve"})).toBeEnabled();
});

test("Ultimate consumes an evacuation checkpoint and exercises compatible and incompatible bridges", async ({page}) => {
  await installScriptedWorker(page, "campaign");
  await enterGame(page);
  await page.getByRole("button", {name: "Solve"}).click();
  await expect.poll(() => page.evaluate(() => window.__workerMessages.some(
    message => message.handoffStage === "packing",
  ))).toBe(true);
  await expect.poll(() => page.evaluate(() => window.__workerMessages.filter(
    message => message.handoffStage === "bridge",
  ).length)).toBeGreaterThanOrEqual(2);
  await expect(page.locator("#search-log-text")).toHaveValue(/Candidate landmark bridges queued/);
  await expect(page.locator("#search-log-text")).toHaveValue(/reason=target-incompatible/);
  await expect(page.locator("#search-log-text")).toHaveValue(/worker released/);
  await expect(page.locator("#search-log-text")).toHaveValue(/firstMessageMs=/);
  await expect(page.locator("#search-log-text")).toHaveValue(/terminateCallMs=/);
});

test("Ultimate gives the structural planner exclusive direct capacity during its head start", async ({page}) => {
  await installScriptedWorker(page, "priority");
  await enterGame(page);
  await page.getByRole("button", {name: "Solve"}).click();
  await expect.poll(() => page.evaluate(() => window.__workerMessages.filter(
    message => message.side === "direct",
  ).length)).toBe(1);
  expect(await page.evaluate(() => window.__workerMessages.filter(
    message => message.side === "direct",
  )[0].handoffStage)).toBe("structural");
  await expect.poll(() => page.evaluate(() => window.__workerMessages.filter(
    message => message.side === "direct",
  ).length)).toBeGreaterThan(1);
});

test("Ultimate publishes its first solution and continues with an exact rewrite", async ({page}) => {
  await installScriptedWorker(page, "anytime");
  await enterGame(page);
  await page.getByRole("button", {name: "Solve"}).click();
  await expect.poll(() => page.evaluate(() => window.__workerMessages.some(
    message => message.algorithm === "solution-window-rewrite",
  ))).toBe(true);
  const rewrite = await page.evaluate(() => window.__workerMessages.find(
    message => message.algorithm === "solution-window-rewrite",
  ));
  expect(rewrite.state.robot).toEqual([1, 2]);
  expect(rewrite.upperBound).toBe(0);
  await expect(page.locator("#search-log-text")).toHaveValue(/replay-validated solution/);
  await expect(page.locator("#search-log-text")).toHaveValue(/replay-validated improvement/);
  await expect(page.locator("#search-log-text")).toHaveValue(/pushes=1 moves=1/);
});
