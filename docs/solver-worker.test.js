const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadWorker(postMessage = () => {}) {
  const source = fs.readFileSync(path.join(__dirname, "solver-worker.js"), "utf8");
  const context = {
    postMessage,
    onmessage: null,
    console,
  };
  vm.runInNewContext(source, context, {filename: "solver-worker.js"});
  return context;
}

function stateFromRows(rows) {
  let robot = null;
  const boxes = [];
  rows.forEach((row, y) => [...row].forEach((cell, x) => {
    if (cell === "R") robot = [y, x];
    if (cell === "X" || (/[A-Z]/.test(cell) && !"ORS".includes(cell))) {
      boxes.push([`${y},${x}`, cell]);
    }
  }));
  return {rows, robot, boxes};
}

const HUGE_ROWS = [
  "OOOOOOOOOOOOOOO", "OaSS   S   SSbO", "OSCS  OOO  SDSO", "OX X  OOO  X XO",
  "O     OOO     O", "OOOO   X   OOOO", "O      O      O", "O G hOOOOOH g O",
  "O      O      O", "OOO         OOO", "OOO   X X   OOO", "OOOOOOOROOOOOOO",
  "O B X X X X A O", "O Sc       dS O", "OOOOOOOOOOOOOOO",
];
const HUGE_SOLUTION =
  "URRLLLLRRDDDRRULDLUUURULLDLURULDLUUUUUUULURRRLDDDDLDDDRDRDRDDDRRRRULLLDLUUULURRDRULDLDDDRRRRRRULLLLL" +
  "DLUUURULLDLURULDLUUUUUUURULLLRRDDDLLLURDRRUUULLDDURRDDLUURULDDLLDRURRDLRDDDDDRRRRRLURRDRURUULRLRDDLL" +
  "DRLRLLLLDDDDLLURDRUUURULLDLURULDLUUUUUUURULDDDDDDDRDRDRDDDLLLLURRRDRUUURULLDLURULDLUUUUUURULDDULLLDR" +
  "URRDDDDDDRRRRRRRUDDLLLULDDDLDRRRRRLLLLUUUUDDDDLLLLLLURRRRRDRUUULULULUURURRRRDRRDDLDLURRUULLRLRDDUULU" +
  "URDDLLULLDLLDDRDRDRRULLDLURULDLLUUURUDLDDDDRRRURDDDRDLLLLLRRRRUUUULLLLUUURUUULLDRURDDRLDLLDRDRLUURDD" +
  "DLDRRRURDDDRDLLLLRRRUUULLLUUUURRURRRRUUUURRRDLLLULDDDRDLLLLLULDDDDDLDRRRURDDDLRLDRRRRLLLUUURRRRLLUUR" +
  "LLDLLULLDLUULURDRUURUULLDRURDLDRRRRRDRUUUULURRRDDDRUDLLUULURLDDDDDRDDLUUUUUULURDDDDDDDLDDRUDRUUURULD" +
  "LUUUULURDDDDDDRDLLLLLLDLUULULLRRDRULLLDRRURDLDRRRRRRRDRUULURLUULLLLLLD";
const HUGE_SOLUTION_250 =
  "DDRRULDLUUURRULLLLDLUUUUUUUULURRRLDDDDLDDDDRRRDDDDRRRRULLLDLUUURULLLLDURRRDDDDRRRRRRULLLLLDLUUURULLL" +
  "DLUUUUUUUURULLLRRDDDLLLUDRRRUUULLDDURRDDLUURULDDLLDURRDRDDDDDRRRDDDDLLURDRUUURULLLDLUUUUUUUURLRULDDD" +
  "DDDDDRRRDDDDLLLLURRRDRUUURULLLDLUUUUUUURULDDDDDDDRRRDDDLLLLDLLURRRRRDRUUURUUDLLLDLUUUURRURRRUUUURRRR" +
  "DLLLULDDDDLDRRDUUDDRRULDLDUULUURDDDRDDLLLLLLDLUUDRRRLLLULUURURUULLDRURDLDRRRRRDRUUUULURRRDDDRULLULLD" +
  "DRUULURRLDDRRDLULLDDRUUULURDDDDDDRLDRDDLLUDLULDDDLDRRRRRLLLLUUURRRRULDULULDRLRDLULDDDLDRRRRLLLUUUULL" +
  "DRURDDDRDLLLLLRRRRUUURRUURUUULLLLLLUUUDLLLDRRRURDLDRRRRRURDDDDDRDLLLULDDDRDLLLLRRRUUULLLLURUULULLDDR" +
  "RURDLDRRRRRDURDRUUUUUUULURDDDDDDDLLLLLLLUUULDLDRRURDLDRRRRRRRDRUULURULLLULLLLD";
test("browser worker solves a one-push dedicated-box puzzle", () => {
  const worker = loadWorker();
  const result = worker.search({
    algorithm: "push-astar",
    state: stateFromRows(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]),
  });

  assert.deepEqual(Array.from(result.path), ["Down"]);
  assert.equal(typeof result.visited, "number");
});

test("browser worker prunes static dead-square pushes", () => {
  const worker = loadWorker();
  const board = worker.parse({
    rows: [
      "OOOOOO",
      "O    O",
      "O RX O",
      "O  S O",
      "OOOOOO",
    ],
  });
  const state = {
    robot: [2, 2],
    boxes: [[2, 3, "X"]],
    cost: 0,
  };

  const moves = worker.neighbors(state, board).map(next => next.move);
  assert.equal(worker.staticDead(2, 4, board, "X"), true);
  assert.equal(moves.includes("Right"), false);
});

test("browser worker prunes 2x2 box deadlocks", () => {
  const worker = loadWorker();
  const board = worker.parse({
    rows: [
      "OOOOOO",
      "O    O",
      "O RXXO",
      "O  XOO",
      "O  SSO",
      "OOOOOO",
    ],
  });
  const boxes = [[2, 3, "X"], [2, 4, "X"], [3, 3, "X"]];

  assert.equal(worker.creates2x2Deadlock(boxes, board, [3, 3]), true);
});

test("bidirectional sides emit compatible compact records", () => {
  const rows = ["OOOOO", "O R O", "O A O", "O a O", "OOOOO"];
  const state = stateFromRows(rows);
  const forwardMessages = [], reverseMessages = [];
  const forward = loadWorker(message => forwardMessages.push(message));
  const reverse = loadWorker(message => reverseMessages.push(message));

  forward.bidirectionalSide({mode: "bidir-forward", state});
  reverse.bidirectionalSide({
    mode: "bidir-reverse",
    state,
    reverseShard: {index: 0, count: 1},
  });

  const records = messages => messages
    .filter(message => message.type === "records")
    .flatMap(message => message.records);
  const forwardRecords = records(forwardMessages);
  const reverseIds = new Set(records(reverseMessages).map(record => record.id));
  assert.equal(forwardRecords.some(record => reverseIds.has(record.id)), true);
  assert.equal(forwardRecords.every(record => !("key" in record)), true);
  assert.equal(forwardRecords.every(record => typeof record.segment === "string"), true);
  const landmarks = reverseMessages
    .filter(message => message.type === "landmarks")
    .flatMap(message => message.landmarks);
  assert.ok(landmarks.length > 0);
  const board = reverse.parse(state);
  const initialBoxes = state.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  assert.equal(landmarks.every(landmark => {
    const targetBoxes = landmark.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]);
    return Number.isFinite(reverse.targetLayoutHeuristic(
      initialBoxes, targetBoxes, board, new Map(),
    ));
  }), true);
});

test("Hungarian matching enforces distinct goals and detects Hall deadlocks", () => {
  const worker = loadWorker();
  assert.equal(worker.minimumAssignmentCost([
    [0, 10, 10],
    [0, 1, 10],
    [10, 1, 0],
  ]), 1);
  assert.equal(worker.minimumAssignmentCost([
    [0, Infinity],
    [0, Infinity],
  ]), Infinity);
});

test("player-aware push distances detect one-way chokepoints", () => {
  const worker = loadWorker();
  const board = worker.parse({rows: [
    "OOOOOOO",
    "O S   O",
    "O X   O",
    "OOO OOO",
    "O  S  O",
    "O     O",
    "OOOOOOO",
  ]});
  const geometric = board.pushDistances.get("1,2");
  const aware = worker.playerAwarePushDistances(board, "2,2");

  assert.equal(geometric.has("2,2"), true);
  assert.equal(aware.has("1,2"), false);
  assert.equal(aware.has("4,3"), true);
});

test("topology analysis prefers deeper goals in one-entrance rooms", () => {
  const worker = loadWorker();
  const board = worker.parse({rows: [
    "OOOOOOO",
    "O     O",
    "OOO OOO",
    "O  S  O",
    "O S   O",
    "O     O",
    "OOOOOOO",
  ]});
  const room = board.topology.rooms[0];

  assert.equal(room.gate, "2,3");
  assert.ok(room.depths.get("4,2") > room.depths.get("3,3"));
  assert.ok(
    worker.topologyPenalty([[3, 3, "X"]], board) >
      worker.topologyPenalty([[4, 2, "X"]], board),
  );
});

test("room evacuation pressure is derived from surplus room contents", () => {
  const worker = loadWorker();
  const board = worker.parse({rows: [
    "OOOOOOO",
    "O     O",
    "OOO OOO",
    "O S   O",
    "O     O",
    "O     O",
    "OOOOOOO",
  ]});
  const crowded = [[3, 3, "X"], [4, 4, "X"]];
  const evacuated = [[1, 3, "X"], [4, 4, "X"]];

  assert.ok(
    worker.roomEvacuationPenalty(crowded, board) >
      worker.roomEvacuationPenalty(evacuated, board),
  );
});

test("puzzle analysis builds a board-derived worker plan", () => {
  const worker = loadWorker();
  const analysis = worker.search({
    algorithm: "analyze-puzzle",
    state: stateFromRows(HUGE_ROWS),
  }).analysis;

  assert.equal(analysis.difficulty, "extreme");
  assert.equal(analysis.boxes, 17);
  assert.ok(analysis.legalPushes > 1);
  assert.ok(analysis.rooms.length > 0);
  assert.ok(analysis.surplusBoxes > 0);
  assert.equal(analysis.recommendations.useEvacuation, true);
  assert.equal(analysis.recommendations.useSequenceMacros, true);
  assert.deepEqual(
    Array.from(analysis.phases, phase => phase.id).slice(0, 2),
    ["evacuation", "room-packing"],
  );
});

test("puzzle analysis keeps simple boards on a small portfolio", () => {
  const worker = loadWorker();
  const analysis = worker.search({
    algorithm: "analyze-puzzle",
    state: stateFromRows(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]),
  }).analysis;

  assert.equal(analysis.difficulty, "small");
  assert.equal(analysis.recommendations.beamAttempts, 1);
  assert.equal(analysis.recommendations.useEvacuation, false);
  assert.equal(analysis.phases.at(-1).id, "exact-proof");
});

test("reverse search charges one unit per pull regardless of walking", () => {
  const worker = loadWorker();
  const board = worker.parse({rows: ["OOOOO", "O   O", "O   O", "O a O", "OOOOO"]});
  const state = {robot: [2, 2], boxes: [[3, 2, "A"]], cost: 0};
  const pulls = worker.reversePullNeighbors(state, board);
  assert.equal(pulls.some(next => next.cost === 1), true);
});

test("frozen components are pruned without rejecting movable box groups", () => {
  const worker = loadWorker();
  const frozenBoard = worker.parse({rows: ["OOOOOOO", "O    SO", "OOOOOOO"]});
  const frozenBoxes = [[1, 2, "X"], [1, 3, "X"], [1, 4, "X"]];
  assert.equal(
    worker.createsFrozenComponentDeadlock(frozenBoxes, frozenBoard, [1, 3]),
    true,
  );

  const openBoard = worker.parse({rows: [
    "OOOOOOO",
    "O     O",
    "O    SO",
    "O     O",
    "OOOOOOO",
  ]});
  assert.equal(
    worker.createsFrozenComponentDeadlock(frozenBoxes.map(([y, x, label]) => [y + 1, x, label]), openBoard, [2, 3]),
    false,
  );
});

test("push beam returns a replayable solution", () => {
  const worker = loadWorker();
  const result = worker.search({
    algorithm: "push-beam",
    beamWidth: 20,
    state: stateFromRows(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]),
  });
  assert.deepEqual(Array.from(result.path), ["Down"]);
});

test("bounded beams return replayable checkpoints for worker handoff", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOOOOO",
    "O R X SO",
    "OOOOOOOO",
  ]);
  const result = worker.search({
    algorithm: "push-beam",
    state,
    beamWidth: 4,
    maxVisited: 2,
    forcedMacros: false,
  });

  assert.equal(result.path, null);
  assert.ok(result.checkpoint);
  assert.ok(result.checkpoint.estimate < 3);
  assert.ok(result.checkpoint.path.length > 0);
});

test("forced push macro collapses a globally forced corridor", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "OOROO",
    "OOAOO",
    "OO OO",
    "OOaOO",
    "OOOOO",
  ]);
  const board = worker.parse(state);
  const initial = {
    robot: state.robot,
    boxes: state.boxes.map(([position, label]) => [...position.split(",").map(Number), label]),
  };
  const first = worker.pushNeighbors(initial, board)[0];
  const collapsed = worker.collapseForcedPushes(first, board);

  assert.equal(collapsed.pushes, 2);
  assert.deepEqual(Array.from(collapsed.path), ["Down", "Down"]);
  assert.equal(worker.goal(collapsed.boxes, board.goals), true);
});

test("box-run macros preserve a replayable sequence of pushes", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOOO",
    "O R A aO",
    "OOOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const first = worker.pushNeighbors(state, board)
    .find(candidate => candidate.pushClass.endsWith(":Right"));
  const sequences = worker.expandPushSequences(first, board, 4, 12, 4);
  const solved = sequences.find(sequence => worker.goal(sequence.boxes, board.goals));

  assert.ok(solved);
  assert.equal(solved.pushes, 2);
  assert.deepEqual(Array.from(solved.path), ["Right", "Right", "Right"]);
});

test("beam selection reserves room for heuristic detours and push diversity", () => {
  const worker = loadWorker();
  const candidates = [];
  for (let index = 0; index < 40; index++) {
    const estimate = index < 10 ? 10 : index < 20 ? 14 : index < 30 ? 18 : 25;
    candidates.push({
      exactSignature: `state-${index}`,
      pushClass: `box-${index % 5}`,
      estimate,
      score: estimate * 3 + index / 100,
      exploreScore: index / 100,
    });
  }

  const selected = worker.selectBeamLayer(candidates, 20, "detour");
  const counts = [0, 0, 0, 0];
  selected.forEach(candidate => {
    const slack = candidate.estimate - 10;
    counts[slack <= 2 ? 0 : slack <= 5 ? 1 : slack <= 9 ? 2 : 3]++;
  });
  assert.deepEqual(counts, [6, 5, 5, 4]);
  assert.equal(new Set(selected.map(candidate => candidate.pushClass)).size, 5);
});

test("bounded transposition maps evict old entries", () => {
  const worker = loadWorker();
  const memo = vm.runInContext("new BoundedDepthMap(2)", worker);
  memo.set("a", 1);
  memo.set("b", 2);
  memo.set("c", 3);

  assert.equal(memo.size, 2);
  assert.equal(memo.has("a"), false);
  assert.equal(memo.get("c"), 3);
});

test("beam restarts honor incumbent push bounds", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "OOROO",
    "OOAOO",
    "OO OO",
    "OOaOO",
    "OOOOO",
  ]);
  const tooTight = worker.search({
    algorithm: "push-beam-restarts",
    state,
    beamWidth: 10,
    restartCount: 2,
    restartVisited: 20,
    upperBound: 1,
  });
  const exact = worker.search({
    algorithm: "push-beam-restarts",
    state,
    beamWidth: 10,
    restartCount: 2,
    restartVisited: 20,
    upperBound: 2,
  });

  assert.equal(tooTight.path, null);
  assert.deepEqual(Array.from(exact.path), ["Down", "Down"]);
  assert.equal(exact.restart, 1);
});

test("bounded push DFS finds a solution at the incumbent ceiling", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "OOROO",
    "OOAOO",
    "OO OO",
    "OOaOO",
    "OOOOO",
  ]);
  const tooTight = worker.search({
    algorithm: "bounded-push-dfs",
    state,
    upperBound: 1,
    maxVisited: 100,
  });
  const exact = worker.search({
    algorithm: "bounded-push-dfs",
    state,
    upperBound: 2,
    maxVisited: 100,
    transpositionLimit: 10,
  });

  assert.equal(tooTight.path, null);
  assert.deepEqual(Array.from(exact.path), ["Down", "Down"]);
  assert.ok(exact.retained <= 10);
});

test("bounded push DFS can limit cumulative ordering discrepancies", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "OOROO",
    "OOAOO",
    "OO OO",
    "OOaOO",
    "OOOOO",
  ]);
  const result = worker.search({
    algorithm: "bounded-push-dfs",
    state,
    upperBound: 2,
    maxVisited: 100,
    discrepancyLimit: 0,
  });

  assert.deepEqual(Array.from(result.path), ["Down", "Down"]);
  assert.equal(result.discrepancyLimit, 0);
});

test("bounded push DFS returns its best checkpoint when its contour is incomplete", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOOOOOO",
    "O R X  SO",
    "OOOOOOOOO",
  ]);
  const result = worker.search({
    algorithm: "bounded-push-dfs",
    state,
    upperBound: 3,
    maxVisited: 2,
  });

  assert.equal(result.path, null);
  assert.ok(result.checkpoint);
  assert.ok(result.checkpoint.estimate < 3);
  assert.ok(result.checkpoint.path.length > 0);
});

test("push IDA star finds a solution on its admissible contour", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "OOROO",
    "OOAOO",
    "OO OO",
    "OOaOO",
    "OOOOO",
  ]);
  const result = worker.search({
    algorithm: "push-ida-star",
    state,
    upperBound: 2,
    maxVisited: 100,
    transpositionLimit: 10,
  });

  assert.deepEqual(Array.from(result.path), ["Down", "Down"]);
  assert.ok(result.visited <= 3);
});

test("push IDA star rejects an unreachable assignment without looping at infinity", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "OXR O",
    "O   O",
    "O  SO",
    "OOOOO",
  ]);
  const result = worker.search({
    algorithm: "push-ida-star",
    state,
    upperBound: Infinity,
    maxVisited: 100,
  });

  assert.equal(result.path, null);
  assert.equal(result.cutoff, false);
  assert.equal(result.visited, 0);
});

test("push IDA star honors an unbounded contour over a finite fallback bound", () => {
  const worker = loadWorker();
  const corridor = `ORX${" ".repeat(30)}SO`;
  const state = stateFromRows(["O".repeat(corridor.length), corridor,
    "O".repeat(corridor.length)]);
  const result = worker.search({
    algorithm: "push-ida-star",
    state,
    upperBound: Infinity,
    pushBound: 30,
    maxVisited: 100,
  });

  assert.equal(result.path.length, 31);
  assert.equal(result.path.every(move => move === "Right"), true);
});

test("persistent exact shards partition a contour without losing its solution", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "OOROO",
    "OOAOO",
    "OO OO",
    "OOaOO",
    "OOOOO",
  ]);
  const results = [0, 1].map(index => worker.search({
    algorithm: "push-ida-star",
    state,
    upperBound: Infinity,
    maxVisited: 100,
    exactShard: {index, count: 2, depth: 1},
  }));

  assert.equal(results.filter(result => result.path).length, 1);
  assert.deepEqual(
    Array.from(results.find(result => result.path).path),
    ["Down", "Down"],
  );
  assert.equal(results.every(result => result.exactShard.count === 2), true);
});

test("Huge exact contour distributes useful work across four persistent shards", () => {
  const worker = loadWorker();
  const state = stateFromRows(HUGE_ROWS);
  const results = [0, 1, 2, 3].map(index => worker.search({
    algorithm: "push-ida-star",
    state,
    upperBound: Infinity,
    maxVisited: 200,
    transpositionLimit: 500,
    exactShard: {index, count: 4, depth: 4},
    seed: 911 + index * 104729,
  }));

  assert.equal(results.every(result => result.cutoff), true);
  assert.equal(results.every(result => result.visited === 200), true);
  assert.equal(results.every(result => result.threshold >= 208), true);
});

test("persistent exact progress explains contour and shard pruning", () => {
  const messages = [];
  const worker = loadWorker(message => messages.push(message));
  const result = worker.search({
    algorithm: "push-ida-star",
    state: stateFromRows(HUGE_ROWS),
    upperBound: Infinity,
    maxVisited: 80,
    progressInterval: 20,
    transpositionLimit: 30,
    exactShard: {index: 0, count: 2, depth: 4},
  });
  const progressMessages = messages.filter(message => message.type === "progress");
  const progress = progressMessages[progressMessages.length - 1];

  assert.equal(result.cutoff, true);
  assert.equal(typeof progress.generated, "number");
  assert.equal(typeof progress.thresholdPrunes, "number");
  assert.equal(typeof progress.transpositionPrunes, "number");
  assert.equal(typeof progress.shardRejected, "number");
  assert.equal(typeof progress.maxDepth, "number");
  assert.equal(progress.nextThreshold === undefined ||
    progress.nextThreshold >= progress.threshold, true);
  assert.equal(typeof result.transpositionEvictions, "number");
  assert.equal(typeof result.maxTranspositions, "number");
});

test("bridge A star connects a forward state to a worker-supplied landmark", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "O R O",
    "O X O",
    "O S O",
    "OOOOO",
  ]);
  const board = worker.parse(state);
  const initial = {
    robot: state.robot,
    boxes: state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const target = worker.pushNeighbors(initial, board)
    .find(next => next.pushClass.endsWith(":Down"));
  const targetState = {
    rows: state.rows,
    robot: target.robot,
    boxes: target.boxes.map(([y, x, label]) => [`${y},${x}`, label]),
  };
  const result = worker.search({algorithm: "bridge-astar", state, targetState});

  assert.deepEqual(Array.from(result.path), Array.from(target.path));
  assert.equal(result.terminationReason, "target-reached");
});

test("bridge A star identifies an incompatible landmark before searching", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOO",
    "O R O",
    "O X O",
    "O S O",
    "OOOOO",
  ]);
  const targetState = {
    rows: state.rows,
    robot: state.robot,
    boxes: [["3,2", "A"]],
  };
  const result = worker.search({algorithm: "bridge-astar", state, targetState});

  assert.equal(result.path, null);
  assert.equal(result.visited, 0);
  assert.equal(result.terminationReason, "target-incompatible");
});

test("bounded bridge search returns a replayable continuation checkpoint", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOOOOOO",
    "O R X S O",
    "OOOOOOOOO",
  ]);
  const targetState = {
    rows: state.rows,
    robot: state.robot,
    boxes: [["1,6", "X"]],
  };
  const result = worker.search({
    algorithm: "bridge-astar",
    state,
    targetState,
    maxVisited: 2,
    forcedMacros: false,
  });

  assert.equal(result.path, null);
  assert.equal(result.cutoff, true);
  assert.ok(result.checkpoint);
  assert.equal(result.checkpoint.cost, 1);
  assert.ok(result.checkpoint.estimate < result.initialEstimate);
  assert.ok(result.checkpoint.path.length > 0);
});

test("bidirectional frontier compaction reports bounded memory telemetry", () => {
  const messages = [];
  const worker = loadWorker(message => messages.push(message));
  const state = stateFromRows(HUGE_ROWS);

  worker.bidirectionalSide({
    mode: "bidir-forward",
    state,
    maxVisited: 100,
    frontierLimit: 2,
  });

  const done = messages.find(message => message.type === "done");
  assert.ok(done);
  assert.ok(done.compactions > 0);
  assert.ok(done.frontier <= 4);
  assert.ok(done.retained <= 4);
  assert.ok(done.generated >= done.visited);
  assert.ok(["budget", "exhausted"].includes(done.terminationReason));
});

test("all hard pruning preserves the known Huge solution", () => {
  const worker = loadWorker();
  const parsed = stateFromRows(HUGE_ROWS);
  const board = worker.parse(parsed);
  const dependencies = board.topology.rooms.flatMap(room => room.dependencies)
    .map(pair => Array.from(pair).join(">"));
  assert.ok(dependencies.includes("13,3>13,2"));
  assert.ok(dependencies.includes("13,11>13,12"));
  let state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [...position.split(",").map(Number), label]),
  };
  const signature = boxes => boxes.map(box => box.join(",")).sort().join(";");
  let pushes = 0;
  for (const code of HUGE_SOLUTION) {
    const move = {U: "Up", D: "Down", L: "Left", R: "Right"}[code];
    const before = signature(state.boxes);
    const next = worker.neighbors(state, board).find(candidate => candidate.move === move);
    assert.ok(next, `known solution move ${move} must remain legal`);
    state = next;
    if (signature(state.boxes) !== before) {
      pushes++;
      assert.ok(pushes + worker.heuristic(state.boxes, board) <= 252);
    }
    const reachable = worker.reachablePaths(state, board);
    assert.equal(worker.createsSealedCorralDeadlock(state, board, reachable), false);
  }

  assert.equal(HUGE_SOLUTION.length, 770);
  assert.equal(pushes, 252);
  assert.equal(worker.goal(state.boxes, board.goals), true);
  assert.equal(worker.heuristic(state.boxes, board), 0);
});

test("the improved Huge replay establishes a 250-push incumbent", () => {
  const worker = loadWorker();
  const parsed = stateFromRows(HUGE_ROWS);
  const board = worker.parse(parsed);
  let state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [...position.split(",").map(Number), label]),
  };
  const signature = boxes => boxes.map(box => box.join(",")).sort().join(";");
  let pushes = 0, maximumBound = 0;
  for (const code of HUGE_SOLUTION_250) {
    const move = {U: "Up", D: "Down", L: "Left", R: "Right"}[code];
    const before = signature(state.boxes);
    const next = worker.neighbors(state, board).find(candidate => candidate.move === move);
    assert.ok(next, `improved solution move ${move} must remain legal`);
    state = next;
    if (signature(state.boxes) !== before) {
      pushes++;
      const bound = pushes + worker.heuristic(state.boxes, board);
      maximumBound = Math.max(maximumBound, bound);
      assert.ok(bound <= 250);
    }
  }

  assert.equal(HUGE_SOLUTION_250.length, 678);
  assert.equal(pushes, 250);
  assert.equal(maximumBound, 250);
  assert.equal(worker.goal(state.boxes, board.goals), true);
});
