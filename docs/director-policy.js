(function attachDirectorPolicy(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.SokomindDirectorPolicy = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";

  const DEFAULT_BRIDGE_LIMITS = Object.freeze({
    maxScheduled: 32,
    maxIncompatible: 16,
    maxProductive: 12,
    maxVisited: 1200000,
    maxWorkerMs: 300000,
  });

  function createBridgeCampaignTracker(overrides = {}) {
    const limits = {...DEFAULT_BRIDGE_LIMITS, ...overrides};
    const campaigns = new Map();

    const get = (campaign) => {
      if (!campaigns.has(campaign)) campaigns.set(campaign, {
        scheduled: 0,
        incompatible: 0,
        productive: 0,
        visited: 0,
        workerMs: 0,
      });
      return campaigns.get(campaign);
    };

    const exhaustedReason = (campaign) => {
      const stats = get(campaign);
      if (stats.incompatible >= limits.maxIncompatible) return "incompatible-limit";
      if (stats.productive >= limits.maxProductive) return "productive-limit";
      if (stats.visited >= limits.maxVisited) return "state-limit";
      if (stats.workerMs >= limits.maxWorkerMs) return "time-limit";
      if (stats.scheduled >= limits.maxScheduled) return "probe-limit";
      return null;
    };

    return {
      limits,
      canSchedule: campaign => exhaustedReason(campaign) === null,
      exhaustedReason,
      recordScheduled(campaign) {
        get(campaign).scheduled++;
        return this.snapshot(campaign);
      },
      recordFinished(campaign, result = {}) {
        const stats = get(campaign);
        const incompatible = result.terminationReason === "target-incompatible";
        if (incompatible) stats.incompatible++;
        else stats.productive++;
        stats.visited += Math.max(0, Number(result.visited) || 0);
        stats.workerMs += Math.max(0, Number(result.workerMs) || 0);
        return this.snapshot(campaign);
      },
      snapshot(campaign) {
        return {...get(campaign), exhaustedReason: exhaustedReason(campaign)};
      },
    };
  }

  function createRequiredWorkTracker(initial = 0) {
    let required = Math.max(0, initial);
    let completed = 0;
    return {
      schedule(count = 1) {
        required += Math.max(0, count);
        return this.snapshot();
      },
      retire(count = 1) {
        required = Math.max(completed, required - Math.max(0, count));
        return this.snapshot();
      },
      finish(count = 1) {
        completed = Math.min(required, completed + Math.max(0, count));
        return this.snapshot();
      },
      isComplete: () => completed === required,
      snapshot: () => ({required, completed, remaining: required - completed}),
    };
  }

  function evaluateBridgeContinuation({
    continuation = 0,
    initialEstimate,
    bestEstimate,
    checkpointCost,
  }) {
    if (continuation >= 2) return {promote: false, reason: "continuation-limit"};
    if (![initialEstimate, bestEstimate, checkpointCost].every(Number.isFinite)) {
      return {promote: false, reason: "missing-metrics"};
    }
    const improvement = initialEstimate - bestEstimate;
    const efficiency = improvement / Math.max(1, checkpointCost);
    const nearTarget = bestEstimate <= 12 && improvement >= 1 && checkpointCost <= 40;
    const efficientProgress = improvement >= 4 && efficiency >= 0.25 &&
      checkpointCost <= 80 && bestEstimate <= 30;
    return {
      promote: nearTarget || efficientProgress,
      reason: nearTarget ? "near-target" : efficientProgress ? "efficient-progress" : "weak-progress",
      improvement,
      efficiency,
      projectedLocalCost: checkpointCost + bestEstimate,
    };
  }

  function selectAnytimeCheckpoints(candidates, limit = 2) {
    const ranked = candidates
      .filter(candidate => candidate?.checkpoint?.state &&
        Number.isFinite(candidate.pushCost) &&
        Number.isFinite(candidate.checkpoint.estimate))
      .map(candidate => ({
        ...candidate,
        projectedCost: candidate.pushCost + candidate.checkpoint.estimate,
      }))
      .sort((left, right) => left.projectedCost - right.projectedCost ||
        left.pushCost - right.pushCost ||
        String(left.id).localeCompare(String(right.id)));
    const selected = [];
    const generations = new Set();
    for (const candidate of ranked) {
      if (selected.length >= limit) break;
      if (generations.has(candidate.generation)) continue;
      selected.push(candidate);
      generations.add(candidate.generation);
    }
    for (const candidate of ranked) {
      if (selected.length >= limit) break;
      if (selected.some(existing => existing.id === candidate.id)) continue;
      selected.push(candidate);
    }
    return selected;
  }

  function exactTranspositionLimit(deviceMemory = 4, shardCount = 1) {
    const memory = Number.isFinite(deviceMemory) && deviceMemory > 0 ? deviceMemory : 4;
    const shards = Math.max(1, Math.floor(shardCount) || 1);
    const totalBudget = memory >= 8 ? 640000 : memory >= 4 ? 480000 : 240000;
    return Math.max(120000, Math.min(320000, Math.floor(totalBudget / shards)));
  }

  function directWorkerCapacity(maxWorkers, activeSideWorkers, evacuationActive) {
    const capacity = Math.max(0, Math.floor(maxWorkers) || 0);
    const sideWorkers = Math.max(0, Math.floor(activeSideWorkers) || 0);
    const available = Math.max(0, capacity - sideWorkers);
    return evacuationActive ? Math.min(2, available) : available;
  }

  function portfolioWorkerCapacity(hardwareConcurrency = 2, deviceMemory = 4) {
    const hardware = Math.max(2, Math.floor(hardwareConcurrency) || 2);
    const memory = Number.isFinite(deviceMemory) && deviceMemory > 0 ? deviceMemory : 4;
    const memoryCapacity = memory <= 2 ? 2 : memory <= 4 ? 3 : 4;
    return Math.max(2, Math.min(4, hardware, memoryCapacity));
  }

  function structuralHeadStartMs(hasStructuralPlan, deviceMemory = 4) {
    if (!hasStructuralPlan) return 0;
    const memory = Number.isFinite(deviceMemory) && deviceMemory > 0 ? deviceMemory : 4;
    return memory <= 4 ? 900 : 600;
  }

  function compareSolutionQuality(left, right) {
    if (!left && !right) return 0;
    if (!left) return 1;
    if (!right) return -1;
    return left.pushes - right.pushes || left.moves - right.moves;
  }

  function acceptsIncumbent(candidate, incumbent = null) {
    return Number.isInteger(candidate?.pushes) && candidate.pushes >= 0 &&
      Number.isInteger(candidate?.moves) && candidate.moves >= 0 &&
      compareSolutionQuality(candidate, incumbent) < 0;
  }

  function tightenedWorkerBound(incumbentPushes, prefixPushes = 0) {
    if (!Number.isFinite(incumbentPushes)) return Infinity;
    return Math.max(0, Math.floor(incumbentPushes) - 1 -
      Math.max(0, Math.floor(prefixPushes) || 0));
  }

  return {
    DEFAULT_BRIDGE_LIMITS,
    createBridgeCampaignTracker,
    createRequiredWorkTracker,
    evaluateBridgeContinuation,
    selectAnytimeCheckpoints,
    exactTranspositionLimit,
    directWorkerCapacity,
    portfolioWorkerCapacity,
    structuralHeadStartMs,
    compareSolutionQuality,
    acceptsIncumbent,
    tightenedWorkerBound,
  };
});
