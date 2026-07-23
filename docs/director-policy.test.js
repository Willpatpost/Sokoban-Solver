const assert = require("node:assert/strict");
const test = require("node:test");

const {
  createBridgeCampaignTracker,
  createRequiredWorkTracker,
  evaluateBridgeContinuation,
  selectAnytimeCheckpoints,
  exactTranspositionLimit,
  directWorkerCapacity,
} = require("./director-policy.js");

test("anytime checkpoint selection minimizes projected cost and preserves phase diversity", () => {
  const state = {rows: [], robot: [0, 0], boxes: []};
  const selected = selectAnytimeCheckpoints([
    {id: "packing-best", generation: "packing", pushCost: 232,
      checkpoint: {state, estimate: 28}},
    {id: "packing-second", generation: "packing", pushCost: 233,
      checkpoint: {state, estimate: 27}},
    {id: "evacuation", generation: "evacuation", pushCost: 58,
      checkpoint: {state, estimate: 152}},
    {id: "expensive", generation: "exact", pushCost: 259,
      checkpoint: {state, estimate: 25}},
  ], 2);

  assert.deepEqual(selected.map(candidate => candidate.id), ["evacuation", "packing-best"]);
  assert.deepEqual(selected.map(candidate => candidate.projectedCost), [210, 260]);
});

test("exact transposition capacity grows for a lone shard without unbounded multi-shard memory", () => {
  assert.equal(exactTranspositionLimit(8, 1), 320000);
  assert.equal(exactTranspositionLimit(8, 3), 213333);
  assert.equal(exactTranspositionLimit(4, 3), 160000);
  assert.equal(exactTranspositionLimit(2, 1), 240000);
  assert.equal(exactTranspositionLimit(undefined, 1), 320000);
});

test("evacuation reserves direct capacity while later phases can fill every free lane", () => {
  assert.equal(directWorkerCapacity(4, 1, true), 2);
  assert.equal(directWorkerCapacity(4, 3, true), 1);
  assert.equal(directWorkerCapacity(4, 1, false), 3);
  assert.equal(directWorkerCapacity(2, 3, false), 0);
});

test("opportunistic bridge churn cannot delay completion of required work", () => {
  const required = createRequiredWorkTracker(3);
  const bridges = createBridgeCampaignTracker({maxIncompatible: 2});
  bridges.recordScheduled("bridge-campaign");
  bridges.recordFinished("bridge-campaign", {terminationReason: "target-incompatible"});
  bridges.recordScheduled("bridge-campaign");

  required.finish(3);
  assert.equal(required.isComplete(), true);
  assert.equal(bridges.snapshot("bridge-campaign").scheduled, 2);
});

test("dynamically scheduled handoffs remain required before exact launch", () => {
  const required = createRequiredWorkTracker(2);
  required.finish();
  required.schedule(2);
  assert.deepEqual(required.snapshot(), {required: 4, completed: 1, remaining: 3});
  required.finish(3);
  assert.equal(required.isComplete(), true);
});

test("bridge campaign stops replacing incompatible targets", () => {
  const tracker = createBridgeCampaignTracker({maxIncompatible: 3, maxScheduled: 20});
  for (let index = 0; index < 3; index++) {
    assert.equal(tracker.canSchedule("packing-a"), true);
    tracker.recordScheduled("packing-a");
    tracker.recordFinished("packing-a", {terminationReason: "target-incompatible"});
  }
  assert.equal(tracker.canSchedule("packing-a"), false);
  assert.equal(tracker.exhaustedReason("packing-a"), "incompatible-limit");
  assert.deepEqual(tracker.snapshot("packing-a"), {
    scheduled: 3,
    incompatible: 3,
    productive: 0,
    visited: 0,
    workerMs: 0,
    exhaustedReason: "incompatible-limit",
  });
});

test("bridge campaigns have independent budgets per checkpoint generation", () => {
  const tracker = createBridgeCampaignTracker({maxScheduled: 1});
  tracker.recordScheduled("reverse-1|packing|checkpoint-a");
  assert.equal(tracker.canSchedule("reverse-1|packing|checkpoint-a"), false);
  assert.equal(tracker.canSchedule("reverse-1|packing|checkpoint-b"), true);
});

test("bridge campaign state and time budgets are hard circuit breakers", () => {
  const states = createBridgeCampaignTracker({maxVisited: 100});
  states.recordScheduled("states");
  states.recordFinished("states", {visited: 100, workerMs: 5});
  assert.equal(states.exhaustedReason("states"), "state-limit");

  const time = createBridgeCampaignTracker({maxWorkerMs: 100});
  time.recordScheduled("time");
  time.recordFinished("time", {visited: 1, workerMs: 100});
  assert.equal(time.exhaustedReason("time"), "time-limit");
});

test("continuations require efficient progress or a credible near-target state", () => {
  assert.equal(evaluateBridgeContinuation({
    initialEstimate: 152, bestEstimate: 44, checkpointCost: 122,
  }).promote, false);
  assert.equal(evaluateBridgeContinuation({
    initialEstimate: 44, bestEstimate: 44, checkpointCost: 2,
  }).promote, false);
  assert.equal(evaluateBridgeContinuation({
    initialEstimate: 18, bestEstimate: 10, checkpointCost: 34,
  }).reason, "near-target");
  assert.equal(evaluateBridgeContinuation({
    continuation: 2, initialEstimate: 20, bestEstimate: 1, checkpointCost: 3,
  }).reason, "continuation-limit");
});
