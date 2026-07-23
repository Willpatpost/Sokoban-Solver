const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const assignmentProfile = require("../bench/assignment-crossover.json");
const localReasoningBaseline = require("../bench/local-reasoning-baseline.json");
const {mirrorRows, rotateRows} = require("../bench/generated-cases.js");

function loadWorker(postMessage = () => {}) {
  const source = ["solver-engine.js", "solver-search.js"]
    .map(file => fs.readFileSync(path.join(__dirname, file), "utf8"))
    .join("\n");
  const context = {
    postMessage,
    onmessage: null,
    console,
  };
  vm.runInNewContext(source, context, {filename: "solver-engine.js"});
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

test("local proof limits match reviewed deterministic cost gates", () => {
  const worker = loadWorker();
  const {schemaVersion, reviewed, build, ...limits} = localReasoningBaseline;
  assert.equal(schemaVersion, 1);
  assert.ok(reviewed);
  assert.ok(build);
  assert.deepEqual({...worker.localReasoningLimits()}, limits);
});

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

test("closed diagonals require wall ends, multiple boxes, and no goal escape", () => {
  const worker = loadWorker();
  const rows = [
    "OOOOOOOO",
    "O O    O",
    "O X O  O",
    "O  O X O",
    "O    O O",
    "O RSS  O",
    "OOOOOOOO",
  ];
  const board = worker.parse({rows});
  const boxes = [[2, 2, "X"], [3, 5, "X"]];
  assert.equal(worker.createsClosedDiagonalDeadlock(boxes, board, [2, 2]), true);

  const escapedRows = [...rows];
  escapedRows[2] = "O XS O  O";
  escapedRows[5] = "O R S  O";
  const escaped = worker.parse({rows: escapedRows});
  assert.equal(worker.createsClosedDiagonalDeadlock(boxes, escaped, [2, 2]), false);

  const typedGoalSequence = worker.parse({rows: [
    "OOOOOOOO", "O O    O", "O a O  O", "O  O B O",
    "O    O O", "O R b  O", "OOOOOOOO",
  ]});
  const typedBoxes = [[2, 2, "A"], [3, 5, "B"]];
  assert.equal(
    worker.createsClosedDiagonalDeadlock(typedBoxes, typedGoalSequence, [2, 2]),
    true,
  );
  const fullyPacked = worker.parse({rows: [
    "OOOOOOOO", "O O    O", "O a O  O", "O  O b O",
    "O    O O", "O R    O", "OOOOOOOO",
  ]});
  assert.equal(
    worker.createsClosedDiagonalDeadlock(typedBoxes, fullyPacked, [2, 2]),
    false,
  );

  const boxEnded = worker.parse({rows: [
    "OOOOOOOO", "OOX    O", "O X O  O", "O  O X O",
    "O    O O", "O RSSS O", "OOOOOOOO",
  ]});
  assert.equal(
    worker.createsClosedDiagonalDeadlock(
      [[1, 2, "X"], [2, 2, "X"], [3, 5, "X"]],
      boxEnded,
      [2, 2],
    ),
    true,
  );
});

test("local pattern tables detect typed corridor order conflicts but preserve bypasses", () => {
  const worker = loadWorker();
  const trapped = stateFromRows([
    "OOOOOOOOO", "OR A BbaO", "OOOOOOOOO",
  ]);
  const trappedBoard = worker.parse(trapped);
  const trappedBoxes = trapped.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  assert.equal(worker.creates2x2Deadlock(trappedBoxes, trappedBoard, [1, 5]), false);
  assert.equal(worker.createsFrozenComponentDeadlock(trappedBoxes, trappedBoard, [1, 5]), false);
  assert.equal(worker.createsPatternDatabaseDeadlock(
    trappedBoxes, trappedBoard, [1, 5],
  ), true);

  const bypass = stateFromRows([
    "OOOOOOOOO", "O       O", "OR A BbaO", "O       O", "O       O",
    "OOOOOOOOO",
  ]);
  const bypassBoard = worker.parse(bypass);
  const bypassBoxes = bypass.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  assert.equal(worker.createsPatternDatabaseDeadlock(
    bypassBoxes, bypassBoard, [2, 5], 256,
  ), false);

  const generic = stateFromRows(["OOOOOOOOO", "OR SS XXO", "OOOOOOOOO"]);
  const genericBoard = worker.parse(generic);
  const genericBoxes = generic.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  assert.equal(worker.createsPatternDatabaseDeadlock(
    genericBoxes, genericBoard, [1, 6],
  ), true);
  const cacheHits = genericBoard.metrics.patternDeadlockCacheHits;
  assert.equal(worker.createsPatternDatabaseDeadlock(
    genericBoxes, genericBoard, [1, 6],
  ), true);
  assert.equal(genericBoard.metrics.patternDeadlockCacheHits, cacheHits + 1);
  assert.ok(genericBoard.metrics.patternCanonicalizations > 0);
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

test("one-row Hungarian repair exactly matches full recomputation", () => {
  const worker = loadWorker();
  assert.equal(
    worker.incrementalAssignmentCrossover(),
    assignmentProfile.javascript.crossover,
  );
  let seed = 0x51f15e;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 0x100000000;
  };
  for (let size = 1; size <= 7; size++) {
    for (let sample = 0; sample < 120; sample++) {
      const costs = Array.from({length: size}, (_, row) =>
        Array.from({length: size}, (_, column) =>
          row === column || random() > 0.12 ? Math.floor(random() * 30) : Infinity));
      const previous = worker.minimumAssignment(costs);
      const changedRow = Math.floor(random() * size);
      const changed = costs.map(row => [...row]);
      changed[changedRow] = Array.from({length: size}, (_, column) =>
        column === changedRow || random() > 0.12 ? Math.floor(random() * 30) : Infinity);
      const repaired = worker.repairMinimumAssignment(previous, changed, changedRow);
      assert.ok(repaired);
      assert.equal(
        repaired.cost,
        worker.minimumAssignmentCost(changed),
        `size=${size} sample=${sample} row=${changedRow}`,
      );
    }
  }
});

test("successor heuristics reuse unchanged assignment rows", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOOOO", "O R     O", "O XXXXX O", "O       O", "O SSSSS O", "OOOOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  assert.ok(Number.isFinite(worker.heuristic(state.boxes, board)));
  const next = worker.pushNeighbors(state, board)[0];
  assert.ok(next);
  assert.ok(Number.isFinite(worker.heuristic(next.boxes, board)));
  assert.equal(board.metrics.incrementalAssignmentCalls, 1);
  assert.equal(board.metrics.incrementalAssignmentFallbacks, 0);
  assert.equal(board.metrics.incrementalAssignmentRowsReused, 4);
});

test("goal commitment oracle distinguishes temporary, conditional, and proven placements", () => {
  const worker = loadWorker();

  const conditionalBoard = worker.parse(stateFromRows([
    "OOOOO", "O R O", "O X O", "O S O", "OOOOO",
  ]));
  assert.equal(
    worker.goalCommitments([[3, 2, "X"]], conditionalBoard).get("3,2"),
    "conditional",
  );

  const provenBoard = worker.parse(stateFromRows([
    "OOOOOO", "O RXSO", "O    O", "OOOOOO",
  ]));
  assert.equal(
    worker.goalCommitments([[1, 4, "X"]], provenBoard).get("1,4"),
    "proven",
  );

  const temporaryBoard = worker.parse(stateFromRows([
    "OOOOOOO", "O R   O", "O XX  O", "O SS  O", "OOOOOOO",
  ]));
  temporaryBoard.topology.rooms = [{
    gate: "1,1",
    cells: new Set(["3,2", "3,3"]),
    goals: ["3,2", "3,3"],
    dependencies: [["3,2", "3,3"]],
  }];
  assert.equal(
    worker.goalCommitments([[3, 2, "X"], [2, 3, "X"]], temporaryBoard).get("3,2"),
    "temporary",
  );

  const corridor = {
    floor: new Set(["1,1", "1,2", "1,3", "1,4", "1,5", "1,6", "1,7"]),
    goals: new Map([["1,5", "X"], ["1,6", "X"]]),
  };
  assert.equal(
    worker.residualMatchingSurvives(
      [[1, 5, "X"], [1, 2, "X"]], corridor, 0, "1,5",
    ),
    false,
  );
});

test("goal packing reserves its strongest reward for proven commitments", () => {
  const worker = loadWorker();
  const board = worker.parse(stateFromRows([
    "OOOOOO", "O RXSO", "O    O", "OOOOOO",
  ]));
  const proven = [[1, 4, "X"]];
  board.commitmentMemo.set(worker.boxSignature(proven, board), new Map([["1,4", "proven"]]));
  const provenBonus = worker.goalPackingBonus(proven, board);
  board.commitmentMemo.set(worker.boxSignature(proven, board), new Map([["1,4", "conditional"]]));
  const conditionalBonus = worker.goalPackingBonus(proven, board);
  board.commitmentMemo.set(worker.boxSignature(proven, board), new Map([["1,4", "temporary"]]));
  const temporaryBonus = worker.goalPackingBonus(proven, board);

  assert.equal(provenBonus, 4 * conditionalBonus);
  assert.equal(temporaryBonus, 0);
});

test("goal commitment integrates support, doorway, and exact room evidence conservatively", () => {
  const worker = loadWorker();
  const conditionalBoard = worker.parse(stateFromRows([
    "OOOOO", "O R O", "O X O", "O S O", "OOOOO",
  ]));
  const placed = [[3, 2, "X"]];
  assert.equal(
    worker.goalCommitments(placed, conditionalBoard, {
      supportDependency: {
        supportDemand: new Map([["3,2", 1]]),
        prerequisiteDemand: new Map(),
      },
    }).get("3,2"),
    "temporary",
  );
  assert.equal(
    worker.goalCommitments(placed, conditionalBoard, {
      supportDependency: {
        supportDemand: new Map(),
        prerequisiteDemand: new Map(),
        prerequisiteClosure: new Map(),
        cycles: new Set(),
        assignmentComplete: true,
      },
      localAnalyses: [{
        kind: "corral",
        proofComplete: true,
        provenCommitments: new Set(["3,2"]),
      }],
    }).get("3,2"),
    "proven",
  );
  assert.equal(
    worker.goalCommitments(placed, conditionalBoard, {
      doorway: {rooms: [{
        index: 0,
        importTotal: 1,
        exportTotal: 0,
        contradictions: [],
        gateLabel: null,
        room: {
          gate: "1,1",
          exteriorStaging: new Set(["3,2"]),
          interiorStaging: new Set(),
        },
      }]},
    }).get("3,2"),
    "temporary",
  );

  const parsed = stateFromRows([
    "OOOOOOO", "O R   O", "OOO OOO", "O  S  O",
    "O  X  O", "O     O", "OOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const state = {robot: parsed.robot, boxes: [[4, 3, "X"]]};
  const room = board.topology.rooms.find(candidate => candidate.goals.includes("3,3"));
  const analysis = worker.exactLocalRoomSearch(state, board, room);
  const packed = [[3, 3, "X"]];
  assert.equal(
    worker.goalCommitments(packed, board, {
      doorway: worker.typedDoorwayFlow(packed, board),
      supportDependency: {supportDemand: new Map(), prerequisiteDemand: new Map()},
      localAnalyses: [analysis],
      transition: {pushedFrom: "4,3", pushedTo: "3,3", pushes: 1},
    }).get("3,3"),
    "proven",
  );
  assert.equal(
    worker.goalCommitments(packed, board, {
      doorway: worker.typedDoorwayFlow(packed, board),
      supportDependency: {
        supportDemand: new Map(),
        prerequisiteDemand: new Map(),
        prerequisiteClosure: new Map(),
        cycles: new Set(),
        assignmentComplete: false,
      },
      localAnalyses: [analysis],
      transition: {pushedFrom: "4,3", pushedTo: "3,3", pushes: 1},
    }).get("3,3"),
    "conditional",
  );
});

test("state-complete commitment locks only boxes proven packed in a finished room", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOO", "O R   O", "OOO OOO", "O  S  O",
    "O     O", "O     O", "OOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const packedState = {robot: [4, 3], boxes: [[3, 3, "X"]]};
  const reachable = worker.reachablePaths(packedState, board);
  const commitments = worker.stateGoalCommitments(packedState, board, reachable);
  assert.equal(commitments.get("3,3"), "proven");
  assert.ok(worker.pushNeighbors(packedState, board, reachable).length > 0);
  assert.equal(
    worker.pushNeighbors(packedState, board, reachable, {lockProven: true}).length,
    0,
  );
  assert.equal(worker.stateGoalCommitments(packedState, board, reachable), commitments);
  assert.ok(board.metrics.commitmentBoxLocks > 0);

  const openBoard = worker.parse({rows: [
    "OOOOOOO", "O     O", "O  R  O", "O  S  O",
    "O     O", "O     O", "OOOOOOO",
  ]});
  const conditionalState = {robot: [2, 3], boxes: [[3, 3, "X"]]};
  const openReachable = worker.reachablePaths(conditionalState, openBoard);
  assert.equal(
    worker.stateGoalCommitments(conditionalState, openBoard, openReachable).get("3,3"),
    "conditional",
  );
  assert.equal(
    worker.pushNeighbors(conditionalState, openBoard, openReachable, {lockProven: true}).length,
    worker.pushNeighbors(conditionalState, openBoard, openReachable).length,
  );
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

test("compiled single-box graph matches the reference search on medium and Huge samples", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOO", "Oa   bO", "O AXB O", "O XRX O",
    "OSCXDSO", "OcS SdO", "OOOOOOO",
  ]);
  const board = worker.parse(parsed);
  for (const start of board.floor) {
    const compiled = [...worker.playerAwarePushDistances(board, start)].sort();
    const reference = [...worker.playerAwarePushDistancesReference(board.floor, start)].sort();
    assert.deepEqual(compiled, reference, `distance mismatch from ${start}`);
  }
  const hugeBoard = worker.parse(stateFromRows(HUGE_ROWS));
  const hugeFloor = [...hugeBoard.floor];
  const sampleStride = Math.max(1, Math.floor(hugeFloor.length / 8));
  for (let index = 0; index < hugeFloor.length; index += sampleStride) {
    const start = hugeFloor[index];
    const compiled = [...worker.playerAwarePushDistances(hugeBoard, start)].sort();
    const reference = [...worker.playerAwarePushDistancesReference(hugeBoard.floor, start)].sort();
    assert.deepEqual(compiled, reference, `Huge distance mismatch from ${start}`);
  }
});

test("typed reverse goal tables match player-aware searches on transformed boards", () => {
  const worker = loadWorker();
  const source = ["OOOOOOO", "Oa   SO", "O A X O", "O  R  O", "OOOOOOO"];
  const boards = [source, mirrorRows(source), rotateRows(source)];
  for (const rows of boards) {
    const board = worker.parse(stateFromRows(rows));
    for (const [goal, label] of board.goals) {
      assert.ok(
        board.goalPushTables.byLabel.get(label).some(table => table.goal === goal),
        `${label} omitted ${goal}`,
      );
      for (const start of board.floor) {
        const reference = worker.playerAwarePushDistancesReference(board.floor, start);
        assert.equal(
          worker.compiledGoalPushDistance(board, start, goal),
          reference.get(goal) ?? Infinity,
          `${start} -> ${goal}`,
        );
      }
    }
    for (const label of ["A", "X"]) {
      assert.deepEqual(
        [...board.goalPushTables.byLabel.get(label)].map(table => table.goal).sort(),
        [...board.goals]
          .filter(([, goalLabel]) => goalLabel === label)
          .map(([goal]) => goal)
          .sort(),
      );
    }
  }
});

test("dense board reachability preserves reference regions and exact walking paths", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOO", "Oa   bO", "O AXB O", "O XRX O",
    "OSCXDSO", "OcS SdO", "OOOOOOO",
  ]);
  const board = worker.parse(parsed);
  let state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [...position.split(",").map(Number), label]),
  };
  const states = [state];
  for (let depth = 0; depth < 3; depth++) {
    const next = worker.neighbors(state, board);
    if (!next.length) break;
    state = next[next.length - 1];
    states.push(state);
  }

  for (const candidate of states) {
    const dense = worker.reachablePaths(candidate, board);
    const reference = worker.reachablePathsReference(candidate, board);
    assert.deepEqual([...dense.keys()].sort(), [...reference.keys()].sort());
    for (const position of reference.keys()) {
      assert.deepEqual(dense.get(position), reference.get(position), `path mismatch at ${position}`);
    }
  }

  assert.equal(board.dense.keys.length, board.floor.size);
  for (const [position, id] of board.dense.idByKey) {
    assert.equal(board.dense.keys[id], position);
  }
});

test("successors derive immutable dense layouts and occupancy indices", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOO", "O R  O", "O AX O", "O aS O", "OOOOOO",
  ]);
  const board = worker.parse(parsed);
  const state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [...position.split(",").map(Number), label]),
  };
  const parentLayout = worker.denseBoxLayout(state.boxes, board);
  const child = worker.pushNeighbors(state, board)[0];
  assert.ok(child);
  const childLayout = worker.denseBoxLayout(child.boxes, board);
  assert.notEqual(childLayout.cells, parentLayout.cells);
  assert.notEqual(childLayout.indexByCell, parentLayout.indexByCell);
  assert.equal(parentLayout.indexByCell[parentLayout.cells[0]], 0);
  childLayout.cells.forEach((cell, index) => {
    assert.equal(childLayout.indexByCell[cell], index);
    assert.notEqual(childLayout.occupancyBits[cell >>> 5] & (1 << (cell & 31)), 0);
  });
  assert.equal(
    worker.boxSignature(child.boxes, board),
    worker.boxSignature(child.boxes.map(box => [...box]), board),
  );
  assert.equal(
    worker.packedBoxIdentity(child.boxes, board),
    worker.packedBoxIdentity(child.boxes.map(box => [...box]), board),
  );
  assert.ok(board.metrics.denseLayoutDerivations > 0);
  assert.ok(board.metrics.denseIdentityUpdates > 0);
  assert.ok(board.metrics.occupancyWordCopies > 0);
});

test("dynamic support dependencies identify blocker routes and enabling pushes", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOO", "O  S  O", "O  X  O", "O  X  O", "OR S  O", "OOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const reachable = worker.reachablePaths(state, board);
  const graph = worker.supportDependencyGraph(state, board, reachable);
  const upperBox = graph.nodes.find(node => node.box === "2,3");

  assert.equal(upperBox.available, false);
  assert.equal(upperBox.options[0].support, "3,3");
  assert.deepEqual(Array.from(upperBox.options[0].blockers), ["3,3"]);
  assert.ok(graph.prerequisiteDemand.get("3,3") > 0);
  assert.equal(graph.assignmentComplete, true);
  assert.ok(graph.prerequisiteEdges.get("2,3").has("3,3"));
  assert.ok(graph.prerequisiteClosure.get("2,3").has("3,3"));
  assert.ok(graph.cycles.has("2,3"));
  assert.ok(graph.cycles.has("3,3"));
  assert.ok(worker.supportDependencyDelta(
    graph, {pushedFrom: "3,3", pushedTo: "3,4"},
  ) < 0);
  assert.ok(worker.supportDependencyDelta(
    graph, {pushedFrom: "4,4", pushedTo: "3,3"},
  ) > 0);

  const cached = worker.supportDependencyGraph(state, board, reachable);
  assert.equal(cached, graph);
  assert.equal(board.metrics.supportDependencyCacheHits, 1);
});

test("exact local room search packs small rooms and exposes optimal first pushes", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOO", "O R   O", "OOO OOO", "O  S  O",
    "O  X  O", "O     O", "OOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const reachable = worker.reachablePaths(state, board);
  const room = board.topology.rooms.find(candidate => candidate.goals.includes("3,3"));
  const result = worker.exactLocalRoomSearch(state, board, room, reachable);

  assert.equal(result.status, "solvable");
  assert.equal(result.pushes, 1);
  assert.equal(result.proofComplete, true);
  assert.equal(result.storage, "dense-bitset");
  assert.ok(result.stateUpperBound > 0);
  assert.ok(result.decompositionComponents >= 1);
  assert.equal(result.importsRequired, 0);
  assert.equal(result.exportsRequired, 0);
  assert.ok(result.viableBoundaries > 0);
  assert.deepEqual(Array.from(result.firstPushes), ["4,3>3,3"]);
  assert.ok(worker.localRoomOrderingDelta(
    [result], {pushedFrom: "4,3", pushedTo: "3,3"},
  ) < 0);
  assert.ok(worker.localRoomOrderingDelta(
    [result], {pushedFrom: "5,2", pushedTo: "5,3"},
  ) > 0);

  assert.equal(worker.exactLocalRoomSearch(state, board, room, reachable), result);
  assert.equal(board.metrics.localRoomCacheHits, 1);
});

test("reverse goal-room packing tables preserve typed labels and optimal doorway choices", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOOOOOOOO", "O R         O", "OOOOOO OOOOOO", "OOOOa b OOOOO",
    "OOOOA B OOOOO", "OOOO    OOOOO", "OOOOOOOOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const state = {
    robot: parsed.robot,
    boxes: parsed.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const room = board.topology.rooms.find(candidate =>
    candidate.goals.includes("3,4") && candidate.goals.includes("3,6"));
  assert.ok(room);
  assert.equal(room.cells.size, 13);
  assert.equal(board.topology.rooms.indexOf(room) >= 0, true);
  const table = worker.reverseGoalRoomPackingTable(board, room);
  assert.equal(table.status, "ready");
  assert.ok(table.states.size > 0);

  const result = worker.exactLocalRoomSearch(state, board, room);
  assert.equal(result.source, "reverse-packing-table");
  assert.equal(result.status, "solvable");
  assert.equal(result.pushes, 2);
  assert.deepEqual(
    [...result.firstPushes].sort(),
    ["4,4>3,4", "4,6>3,6"],
  );
});

test("room pattern tables add only proven multi-box interaction cost", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOOOOOOOO", "O R AB      O", "OOOOOO OOOOOO", "OOOOa b OOOOO",
    "OOOO    OOOOO", "OOOO    OOOOO", "OOOOOOOOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const room = board.topology.rooms.find(candidate => candidate.goals.length === 2);
  const boxes = [[3, 5, "A"], [3, 6, "B"]];
  const table = worker.reverseRoomPatternTable(board, room);
  assert.equal(table.status, "ready");
  assert.equal(table.states.get("3,5,A;3,6,B"), 3);

  const independent = boxes.reduce((total, [y, x, label]) => {
    const goal = board.goalsByLabel.get(label)[0];
    return total + worker.playerAwarePushDistances(board, `${y},${x}`).get(goal);
  }, 0);
  assert.equal(independent, 1);
  assert.equal(worker.heuristic(boxes, board), 3);
  assert.equal(board.metrics.roomPatternBoost, 2);

  const exact = worker.exactLocalPushSearch({
    domain: board.floor,
    boxes: boxes.map(([y, x, label]) => [`${y},${x}`, label]),
    robot: "1,2",
    gate: room.gate,
    isGoal: occupied => room.goals.every(goal =>
      occupied.get(goal)?.label === board.goals.get(goal)),
  });
  assert.equal(exact.pushes, 3);

  const preparedBoard = structuredClone(worker.createPreparedBoardSeed(board));
  const hydrated = worker.parse({...parsed, preparedBoard});
  const hydratedRoom = hydrated.topology.rooms.find(candidate => candidate.goals.length === 2);
  assert.equal(
    worker.reverseRoomPatternTable(hydrated, hydratedRoom).states.get("3,5,A;3,6,B"),
    3,
  );
  assert.equal(hydrated.metrics.roomPatternBuilds, 0);
});

test("room pattern bounds minimize shared-label boxes across inside and outside goals", () => {
  const worker = loadWorker();
  const board = worker.parse(stateFromRows([
    "OOOOOOOOOOOOO", "OaR AAB     O", "OOOOOO OOOOOO", "OOOOa b OOOOO",
    "OOOO    OOOOO", "OOOO    OOOOO", "OOOOOOOOOOOOO",
  ]));
  const room = board.topology.rooms.find(candidate => candidate.goals.length === 2);
  const table = worker.reverseRoomPatternTable(board, room);
  assert.equal(table.status, "ready");
  assert.equal(table.targetBoxes.length, 2);
  const boxes = [[1, 4, "A"], [1, 5, "A"], [1, 6, "B"]];
  const assignmentCosts = new Map([
    ["A", worker.minimumAssignment([
      [...board.goalsByLabel.get("A")].map(goal =>
        worker.compiledGoalPushDistance(board, "1,4", goal)),
      [...board.goalsByLabel.get("A")].map(goal =>
        worker.compiledGoalPushDistance(board, "1,5", goal)),
    ]).cost],
    ["B", worker.compiledGoalPushDistance(
      board, "1,6", board.goalsByLabel.get("B")[0])],
  ]);
  const candidates = worker.roomPatternHeuristicCandidates(
    boxes, board, assignmentCosts);
  assert.ok(candidates.every(candidate => Number.isFinite(candidate.boost)));
  assert.equal(board.metrics.patternSelectionCutoffs, 0);
});

test("large rooms partition interaction tables without splitting shared labels", () => {
  const worker = loadWorker();
  const board = worker.parse(stateFromRows([
    "OOOOOOOOOOOOOOOOO",
    "O R             O",
    "OOOOOOOO OOOOOOOO",
    "OOOOOabcde OOOOOO",
    "OOOOOABCDE OOOOOO",
    "OOOOO      OOOOOO",
    "OOOOOOOOOOOOOOOOO",
  ]));
  const room = board.topology.rooms.find(candidate => candidate.goals.length === 5);
  const table = worker.reverseRoomPatternTable(board, room);
  assert.equal(table.status, "partitioned");
  assert.deepEqual(
    Array.from(table.partitions, partition => partition.targetBoxes.length).sort(),
    [2, 3],
  );
  assert.ok(table.partitions.every(partition => partition.status === "ready"));
  const usedLabels = table.partitions.flatMap(partition => [...partition.labels]);
  assert.equal(new Set(usedLabels).size, usedLabels.length);
});

test("pair conflict tables prove extra pushes through a shared chokepoint", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOOOO", "O   O   O", "O R O   O", "Ob A  BaO",
    "O   O   O", "O   O   O", "OOOOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const boxes = [[3, 3, "A"], [3, 6, "B"]];
  const independent = boxes.reduce((total, [y, x, label]) => {
    const goal = board.goalsByLabel.get(label)[0];
    return total + worker.playerAwarePushDistances(board, `${y},${x}`).get(goal);
  }, 0);
  assert.equal(independent, 9);
  assert.equal(worker.heuristic(boxes, board), 11);
  assert.equal(board.metrics.pairConflictBuilds, 1);
  assert.equal(board.metrics.pairConflictHits, 1);
  assert.equal(board.metrics.pairConflictBoost, 2);

  const exact = worker.exactLocalPushSearch({
    domain: board.floor,
    boxes: boxes.map(([y, x, label]) => [`${y},${x}`, label]),
    robot: "2,2",
    gate: "3,4",
    maxStates: 50000,
    isGoal: occupied => [...board.goals].every(([position, label]) =>
      occupied.get(position)?.label === label),
  });
  assert.equal(exact.pushes, 11);

  const preparedBoard = structuredClone(worker.createPreparedBoardSeed(board));
  const hydrated = worker.parse({...parsed, preparedBoard});
  assert.equal(worker.heuristic(boxes, hydrated), 11);
  assert.equal(hydrated.metrics.pairConflictBuilds, 0);
  assert.equal(hydrated.metrics.pairConflictBoost, 2);
});

test("overlapping heuristic conflicts use the maximum label-disjoint combination", () => {
  const worker = loadWorker();
  const candidate = (labels, boost) => ({labels: new Set(labels), boost, kind: "pair"});
  const selected = worker.maximumDisjointPatternSelection([
    candidate(["A", "B"], 3),
    candidate(["B", "C"], 5),
    candidate(["C", "D"], 4),
  ]);
  assert.equal(selected.reduce((total, entry) => total + entry.boost, 0), 7);
  assert.deepEqual(
    Array.from(selected, entry => [...entry.labels].join("")).sort(),
    ["AB", "CD"],
  );
});

test("pair conflict cutoffs fall back while the independent global capacity table remains usable", () => {
  const worker = loadWorker();
  const board = worker.parse(stateFromRows([
    "OOOOOOOOO", "O   O   O", "O R O   O", "Ob A  BaO",
    "O   O   O", "O   O   O", "OOOOOOOOO",
  ]));
  const table = worker.reversePairConflictTable(board, "A", "B", 10);
  assert.equal(table.status, "cutoff");
  assert.equal(table.states.size, 10);
  assert.equal(worker.heuristic([[3, 3, "A"], [3, 6, "B"]], board), 11);
  assert.equal(board.metrics.pairConflictBoost, 0);
  assert.equal(board.metrics.capacityPatternBoost, 2);
});

test("three-box capacity patterns strengthen shared generic assignments safely", () => {
  const worker = loadWorker();
  const parsed = stateFromRows([
    "OOOOOOOOO", "O S S S O", "O       O", "O XXX R O", "O       O",
    "OOOOOOOOO",
  ]);
  const board = worker.parse(parsed);
  const boxes = parsed.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  const assignment = boxes.reduce((total, [y, x]) => {
    const distances = board.goalsByLabel.get("X")
      .map(goal => worker.compiledGoalPushDistance(board, `${y},${x}`, goal));
    return total + Math.min(...distances);
  }, 0);
  const estimate = worker.heuristic(boxes, board);
  const table = worker.reverseCapacityPatternTable(board);
  const relaxed = table.states.get(worker.localBoxSignature(
    boxes.map(([y, x, label]) => [`${y},${x}`, label])));
  assert.equal(table.status, "ready");
  assert.equal(estimate, relaxed);
  assert.ok(estimate >= assignment);
  assert.equal(board.metrics.capacityPatternHits, 1);
  const exact = worker.exactLocalPushSearch({
    domain: board.floor,
    boxes: boxes.map(([y, x, label]) => [`${y},${x}`, label]),
    robot: parsed.robot.join(","),
    gate: parsed.robot.join(","),
    maxStates: 50000,
    isGoal: occupied => [...board.goals].every(([position, label]) =>
      occupied.get(position)?.label === label),
  });
  assert.ok(Number.isFinite(exact.pushes));
  assert.ok(estimate <= exact.pushes);
});

test("goal-cut certificates solve independent components and return a replayable path", () => {
  const worker = loadWorker();
  const rows = [
    "OOOOOOOOOOO",
    "O R A aOOOO",
    "O     OOOOO",
    "OOOOO     O",
    "OOOOOOb B O",
    "OOOOOO    O",
    "OOOOOOOOOOO",
  ];
  for (const transformed of [rows, mirrorRows(rows), rotateRows(rows)]) {
    const state = stateFromRows(transformed);
    const board = worker.parse(state);
    const boxes = state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]);
    const certificate = worker.goalCutDecomposition(boxes, board);
    assert.ok(certificate);
    assert.equal(certificate.components.length, 2);
    const result = worker.search({algorithm: "push-astar", state});
    assert.equal(result.status, "solved");
    assert.equal(result.decompositionComponents, 2);
    assert.ok(result.path.length > 0);
  }
  const incompatible = stateFromRows([
    "OOOOOOOOOOO",
    "O R A bOOOO",
    "O     OOOOO",
    "OOOOO     O",
    "OOOOOOa B O",
    "OOOOOO    O",
    "OOOOOOOOOOO",
  ]);
  const incompatibleBoard = worker.parse(incompatible);
  const incompatibleBoxes = incompatible.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  assert.equal(
    worker.goalCutDecomposition(incompatibleBoxes, incompatibleBoard),
    null,
  );
});

test("exact local room search reports exhausted and import-dependent abstractions safely", () => {
  const worker = loadWorker();
  const blockedParsed = stateFromRows([
    "OOOOOOO", "O R   O", "OOO OOO", "O  S  O",
    "O     O", "OX    O", "OOOOOOO",
  ]);
  const blockedBoard = worker.parse(blockedParsed);
  const blockedState = {
    robot: blockedParsed.robot,
    boxes: [[5, 1, "X"]],
  };
  const blockedRoom = blockedBoard.topology.rooms.find(room => room.goals.includes("3,3"));
  const exhausted = worker.exactLocalRoomSearch(blockedState, blockedBoard, blockedRoom);
  assert.equal(exhausted.status, "exhausted");
  assert.equal(exhausted.proofComplete, true);

  const missingState = {robot: blockedParsed.robot, boxes: []};
  const needsImport = worker.exactLocalRoomSearch(missingState, blockedBoard, blockedRoom);
  assert.equal(needsImport.status, "needs-import");
  assert.equal(needsImport.importsRequired, 1);
  assert.equal(worker.localRoomOrderingDelta(
    [needsImport], {pushedFrom: "1,1", pushedTo: "1,2"},
  ), 0);

  const closedParsed = stateFromRows(["OOOOOOO", "OR SX O", "OOOOOOO"]);
  const closedBoard = worker.parse(closedParsed);
  const closedState = {robot: closedParsed.robot, boxes: [[1, 4, "X"]]};
  const closedRoom = {
    cells: new Set(closedBoard.floor),
    gate: "1,1",
    approach: [],
    goals: ["1,3"],
    dependencies: [],
  };
  closedBoard.topology.rooms = [closedRoom];
  const closed = worker.exactLocalRoomSearch(closedState, closedBoard, closedRoom);
  assert.equal(closed.status, "exhausted");
  assert.equal(closed.proofComplete, true);
  assert.equal(closed.globalDeadlockProven, true);
});

test("exact local corral search finds the push that reopens a small inaccessible region", () => {
  const worker = loadWorker();
  const parsed = stateFromRows(["OOOOOOO", "OR X SO", "OOOOOOO"]);
  const board = worker.parse(parsed);
  const state = {robot: parsed.robot, boxes: [[1, 3, "X"]]};
  const reachable = worker.reachablePaths(state, board);
  const analyses = worker.exactLocalCorralAnalyses(state, board, reachable);

  assert.equal(analyses.length, 1);
  assert.equal(analyses[0].status, "solvable");
  assert.equal(analyses[0].pushes, 1);
  assert.deepEqual(Array.from(analyses[0].firstPushes), ["1,3>1,4"]);
  assert.ok(worker.localRoomOrderingDelta(
    analyses, {pushedFrom: "1,3", pushedTo: "1,4"},
  ) < 0);

  assert.equal(worker.exactLocalCorralAnalyses(state, board, reachable)[0], analyses[0]);
  assert.equal(board.metrics.localCorralCacheHits, 1);

  const closedParsed = stateFromRows(["OOOOOOOOO", "OR  S S O", "OOOOOOOOO"]);
  const closedBoard = worker.parse(closedParsed);
  const closedState = {robot: closedParsed.robot, boxes: [[1, 2, "X"], [1, 6, "X"]]};
  const closedReachable = worker.reachablePaths(closedState, closedBoard);
  const closedAnalysis = worker.exactLocalCorralAnalyses(
    closedState, closedBoard, closedReachable,
  )[0];
  assert.ok(closedAnalysis.provenCommitments.has("1,6"));
  assert.equal(
    worker.stateGoalCommitments(closedState, closedBoard, closedReachable).get("1,6"),
    "temporary",
  );
});

test("compact box signatures are permutation-invariant and collision-free on a small board", () => {
  const worker = loadWorker();
  const board = worker.parse(stateFromRows([
    "OOOOOO", "Oa  bO", "O    O", "O    O", "OOOOOO",
  ]));
  const cells = [...board.floor].map(position => position.split(",").map(Number));
  const signatures = new Map();
  const identities = new Map();
  for (let left = 0; left < cells.length; left++) {
    for (let right = 0; right < cells.length; right++) {
      if (left === right) continue;
      const boxes = [
        [cells[left][0], cells[left][1], "A"],
        [cells[right][0], cells[right][1], "B"],
      ];
      const compact = worker.boxSignature(boxes, board);
      const reference = worker.boxSignatureReference(boxes);
      assert.equal(signatures.get(compact) ?? reference, reference);
      signatures.set(compact, reference);
      assert.equal(worker.boxSignature([...boxes].reverse(), board), compact);
      assert.equal(worker.boxSignature(boxes, board), compact);
      const identity = worker.packedBoxIdentity(boxes, board);
      assert.equal(identities.get(identity) ?? reference, reference);
      identities.set(identity, reference);
      assert.equal(worker.packedBoxIdentity([...boxes].reverse(), board), identity);
      assert.equal(worker.packedBoxIdentity(boxes, board), identity);
    }
  }
  assert.equal(signatures.size, cells.length * (cells.length - 1));
  assert.equal(identities.size, signatures.size);
  assert.ok(board.metrics.signatureCacheHits > 0);
  assert.ok(board.metrics.packedIdentityCacheHits > 0);
});

test("compact canonical push keys preserve robot-region equivalence", () => {
  const worker = loadWorker();
  const board = worker.parse(stateFromRows([
    "OOOOOOO", "O  a  O", "O  A  O", "O     O", "OOOOOOO",
  ]));
  const boxes = [[2, 3, "A"]];
  const left = {robot: [3, 1], boxes};
  const right = {robot: [3, 5], boxes};
  const leftKey = worker.pushKey(left, worker.reachablePaths(left, board));
  const rightKey = worker.pushKey(right, worker.reachablePaths(right, board));
  assert.equal(leftKey, rightKey);
  assert.notEqual(worker.exactPushKey(left, board), worker.exactPushKey(right, board));
  assert.equal(
    worker.pushIdentity(left, worker.reachablePaths(left, board)),
    worker.pushIdentity(right, worker.reachablePaths(right, board)),
  );
  assert.notEqual(worker.exactPushIdentity(left, board), worker.exactPushIdentity(right, board));
});

test("prepared board seeds are clone-safe and preserve search results", () => {
  const worker = loadWorker();
  const state = stateFromRows([
    "OOOOOO", "O R  O", "O AB O", "O ab O", "OOOOOO",
  ]);
  const analysis = worker.search({algorithm: "analyze-puzzle", state}).analysis;
  const preparedBoard = structuredClone(analysis.preparedBoard);
  const firstBoard = worker.parse({...state, preparedBoard});
  const secondBoard = worker.parse({...state, preparedBoard});
  assert.notEqual(firstBoard.heuristicMemo, secondBoard.heuristicMemo);
  assert.notEqual(firstBoard.deadlockMemo, secondBoard.deadlockMemo);
  assert.notEqual(firstBoard.commitmentMemo, secondBoard.commitmentMemo);
  assert.notEqual(firstBoard.stateCommitmentMemo, secondBoard.stateCommitmentMemo);
  assert.notEqual(firstBoard.supportDependencyMemo, secondBoard.supportDependencyMemo);
  assert.notEqual(firstBoard.localRoomMemo, secondBoard.localRoomMemo);
  assert.notEqual(firstBoard.roomPatternTables, secondBoard.roomPatternTables);
  assert.notEqual(firstBoard.pairConflictTables, secondBoard.pairConflictTables);
  assert.notEqual(firstBoard.shortestCorridorMemo, secondBoard.shortestCorridorMemo);
  assert.notEqual(firstBoard.localCorralMemo, secondBoard.localCorralMemo);
  assert.notEqual(firstBoard.doorwayFlowMemo, secondBoard.doorwayFlowMemo);
  assert.notEqual(firstBoard.denseBoxMemo, secondBoard.denseBoxMemo);
  assert.notEqual(firstBoard.boxSignatureMemo, secondBoard.boxSignatureMemo);
  assert.equal(firstBoard.topology, secondBoard.topology);
  assert.equal(firstBoard.goalPushTables, secondBoard.goalPushTables);
  assert.ok(preparedBoard.playerPushDistances.size > 0);
  assert.ok(preparedBoard.estimatedBytes > 0);
  const baseline = worker.search({algorithm: "push-astar", state});
  const reused = worker.search({
    algorithm: "push-astar",
    state: {...state, preparedBoard},
  });

  assert.deepEqual(Array.from(reused.path), Array.from(baseline.path));
  assert.equal(reused.visited, baseline.visited);
  assert.equal(reused.performance.preparedBoardReuses, 1);
  assert.equal(reused.performance.preparedBoardFallbacks, 0);
  assert.equal(reused.performance.graphCompileMs, 0);
  assert.equal(reused.performance.denseBuildMs, 0);
  assert.equal(reused.performance.goalTableMs, 0);
  assert.ok(reused.performance.preparedBoardHydrateMs >= 0);
  assert.equal(
    reused.performance.preparedPlayerDistanceTables,
    preparedBoard.playerPushDistances.size,
  );
  assert.ok(reused.performance.graphNodes > 0);
});

test("prepared board seeds fall back safely when board contents differ", () => {
  const worker = loadWorker();
  const source = stateFromRows(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]);
  const target = stateFromRows(["OOOOOO", "O R  O", "O A  O", "O  a O", "OOOOOO"]);
  const sourceBoard = worker.parse(source);
  const preparedBoard = structuredClone(worker.createPreparedBoardSeed(sourceBoard));
  const result = worker.search({
    algorithm: "push-astar",
    state: {...target, preparedBoard},
  });

  assert.equal(result.performance.preparedBoardReuses, 0);
  assert.equal(result.performance.preparedBoardFallbacks, 1);
  assert.equal(result.performance.denseCells, 12);
});

test("search results expose bounded hot-path performance telemetry", () => {
  const worker = loadWorker();
  const result = worker.search({
    algorithm: "push-beam",
    beamWidth: 20,
    state: stateFromRows(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]),
  });
  assert.ok(result.performance.totalMs >= 0);
  assert.equal(result.performance.heapSupported, false);
  assert.equal(result.performance.heapUsedBytes, null);
  assert.equal(result.performance.heapPeakBytes, null);
  assert.equal(result.performance.heapDeltaBytes, null);
  assert.equal(result.performance.heapSamples, 0);
  assert.ok(result.performance.graphNodes > 0);
  assert.ok(result.performance.graphEdges > 0);
  assert.ok(result.performance.denseCells > 0);
  assert.ok(result.performance.denseBuildMs >= 0);
  assert.ok(result.performance.signatureCalls > 0);
  assert.ok(result.performance.signatureCacheHits > 0);
  assert.ok(result.performance.signatureCharacters > 0);
  assert.ok(result.performance.signatureMs >= 0);
  assert.ok(result.performance.packedIdentityCalls > 0);
  assert.equal(typeof result.performance.packedIdentityCacheHits, "number");
  assert.ok(result.performance.packedIdentityValues > 0);
  assert.equal(result.performance.preparedBoardReuses, 0);
  assert.ok(result.performance.heuristicCalls > 0);
  assert.ok(result.performance.supportDependencyCalls > 0);
  assert.ok(result.performance.supportDependencyMs >= 0);
  assert.ok(result.performance.localRoomMs >= 0);
  assert.equal(typeof result.performance.roomPatternBuilds, "number");
  assert.equal(typeof result.performance.roomPatternStates, "number");
  assert.equal(typeof result.performance.roomPatternHits, "number");
  assert.equal(typeof result.performance.roomPatternBoost, "number");
  assert.equal(typeof result.performance.pairConflictBuilds, "number");
  assert.equal(typeof result.performance.pairConflictStates, "number");
  assert.equal(typeof result.performance.pairConflictCandidates, "number");
  assert.equal(typeof result.performance.pairConflictHits, "number");
  assert.equal(typeof result.performance.pairConflictBoost, "number");
  assert.equal(typeof result.performance.beamFeatureCells, "number");
  assert.equal(typeof result.performance.beamFeatureSelections, "number");
  assert.equal(typeof result.performance.beamBandSelections, "number");
  assert.ok(result.performance.localCorralMs >= 0);
  assert.ok(result.performance.doorwayFlowMs >= 0);
  assert.ok(result.performance.reachabilityCalls > 0);
  assert.ok(result.performance.pushNeighborCalls > 0);
  assert.ok(result.performance.pushesRetained > 0);
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

test("typed doorway flow tracks label direction, lane geometry, and staging capacity", () => {
  const worker = loadWorker();
  const board = worker.parse({rows: [
    "OOOOOOO", "O     O", "O     O", "O     O", "OOO OOO",
    "O  a  O", "O b   O", "O     O", "OOOOOOO",
  ]});
  const room = board.topology.rooms.find(candidate => candidate.goals.includes("5,3"));
  const boxes = [
    [6, 3, "A"], [7, 4, "A"],
    [2, 2, "B"], [2, 4, "B"],
  ];
  const analysis = worker.typedDoorwayFlow(boxes, board);
  const flow = analysis.rooms.find(candidate => candidate.room === room);

  assert.equal(room.gate, "3,3");
  assert.equal(room.doorwayLanes.some(lane => lane.importPossible), true);
  assert.equal(room.doorwayLanes.some(lane => lane.exportPossible), true);
  assert.equal(flow.imports.get("B"), 1);
  assert.equal(flow.exports.get("A"), 1);
  assert.ok(flow.interiorCapacity > 0);
  assert.ok(flow.exteriorCapacity > 0);
  assert.ok(worker.doorwayFlowDelta(analysis, {boxes}, {
    pushedFrom: "2,3", pushedTo: "3,3", pushClass: "B:2,3:Down",
  }) < 0);
  assert.ok(worker.doorwayFlowDelta(analysis, {boxes}, {
    pushedFrom: "4,3", pushedTo: "3,3", pushClass: "A:4,3:Up",
  }) < 0);
  assert.ok(worker.doorwayFlowDelta(analysis, {boxes}, {
    pushedFrom: "2,3", pushedTo: "3,3", pushClass: "A:2,3:Down",
  }) > 0);

  const stagingFull = [
    ...boxes,
    ...[...room.exteriorStaging].map(position => [
      ...position.split(",").map(Number), "X",
    ]),
  ];
  const fullFlow = worker.typedDoorwayFlow(stagingFull, board).rooms
    .find(candidate => candidate.room === room);
  assert.equal(fullFlow.exteriorCapacity, 0);
  assert.ok(fullFlow.contradictions.includes("exterior-staging-full"));

  assert.equal(worker.typedDoorwayFlow(boxes, board), analysis);
  assert.equal(board.metrics.doorwayFlowCacheHits, 1);
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
  assert.equal(analysis.recommendations.reverseWorkerLimit, 2);
  assert.ok(analysis.reverseStartRegions >= 1);
  assert.ok(analysis.productiveReverseStartRegions >= 1);
  assert.ok(analysis.reverseStartPulls >= analysis.productiveReverseStartRegions);
  assert.ok(analysis.preparedBoardStats.estimatedBytes > 0);
  assert.ok(analysis.preparedBoardStats.buildMs >= 0);
  assert.equal(analysis.preparedBoardStats.goalTables, analysis.goals);
  assert.ok(analysis.preparedBoardStats.playerDistanceTables > 0);
  assert.equal(
    analysis.preparedBoardStats.playerDistanceTables,
    analysis.preparedBoard.playerPushDistances.size,
  );
  assert.deepEqual(
    Array.from(analysis.phases, phase => phase.id).slice(0, 2),
    ["evacuation", "room-packing"],
  );
});

test("reverse workers receive disjoint first-pull branches from every solved-side region", () => {
  const worker = loadWorker();
  const state = stateFromRows(HUGE_ROWS);
  const board = worker.parse(state);
  const boxes = state.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  const all = worker.reverseStartStates(board, boxes, {index: 0, count: 1});
  const first = worker.reverseStartStates(board, boxes, {index: 0, count: 2});
  const second = worker.reverseStartStates(board, boxes, {index: 1, count: 2});
  assert.equal(first.length, all.length);
  assert.equal(second.length, all.length);
  assert.equal(first.portfolioStats.totalRegions, all.length);
  assert.equal(second.portfolioStats.totalRegions, all.length);
  assert.equal(
    first.portfolioStats.assignedPullOptions + second.portfolioStats.assignedPullOptions,
    all.portfolioStats.totalPullOptions,
  );
  assert.ok(first.portfolioStats.assignedPullOptions > 0);
  assert.ok(second.portfolioStats.assignedPullOptions > 0);
  const portfolio = worker.reverseStartPortfolio(board, boxes);
  for (const entry of portfolio) {
    for (const signature of entry.pullSignatures) {
      assert.notEqual(
        worker.reverseShardOwns(signature, {index: 0, count: 2}),
        worker.reverseShardOwns(signature, {index: 1, count: 2}),
      );
    }
  }
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

  const recursiveBoard = worker.parse(stateFromRows([
    "OOOOOOOO", "OOX    O", "O X    O", "O R SS O", "OOOOOOOO",
  ]));
  const recursiveBoxes = [[1, 2, "X"], [2, 2, "X"]];
  assert.equal(
    worker.createsFrozenComponentDeadlock(recursiveBoxes, recursiveBoard, [2, 2]),
    true,
  );
  assert.ok(recursiveBoard.metrics.recursiveFreezeBoxes > 0);
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

test("bounded beams report liveness by elapsed time before their state interval", () => {
  const messages = [];
  const worker = loadWorker(message => messages.push(message));
  worker.search({
    algorithm: "push-beam",
    state: stateFromRows([
      "OOOOOOOO",
      "O R X SO",
      "OOOOOOOO",
    ]),
    beamWidth: 4,
    maxVisited: 2,
    progressInterval: 5000,
    progressIntervalMs: -1,
    forcedMacros: false,
  });

  const progress = messages.find(message => message.type === "progress");
  assert.ok(progress);
  assert.ok(progress.visited > 0);
  assert.ok(progress.visited < 5000);
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

  const selected = worker.selectBeamLayer(candidates, 20, "detour", null, false);
  const counts = [0, 0, 0, 0];
  selected.forEach(candidate => {
    const slack = candidate.estimate - 10;
    counts[slack <= 2 ? 0 : slack <= 5 ? 1 : slack <= 9 ? 2 : 3]++;
  });
  assert.deepEqual(counts, [6, 5, 5, 4]);
  assert.equal(new Set(selected.map(candidate => candidate.pushClass)).size, 5);
});

test("beam feature archives retain strategically distinct cells beyond score bands", () => {
  const worker = loadWorker();
  const candidates = [];
  for (let index = 0; index < 30; index++) {
    candidates.push({
      exactSignature: `direct-${index}`,
      pushClass: "same-push",
      estimate: 10,
      score: index,
      exploreScore: index,
      topology: 0,
      evacuation: 0,
      packing: 0,
      doorway: 0,
      doorwayDelta: 0,
      dependencyDelta: 0,
      localRoomDelta: 0,
    });
  }
  const rareFeatures = [
    {topology: 4},
    {evacuation: 2},
    {packing: 2},
    {doorway: 2, doorwayDelta: 1},
    {dependencyDelta: -2},
    {localRoomDelta: 2},
  ];
  rareFeatures.forEach((features, index) => candidates.push({
    exactSignature: `feature-${index}`,
    pushClass: "same-push",
    estimate: 10,
    score: 100 + index,
    exploreScore: 100 + index,
    ...features,
  }));

  const legacy = worker.selectBeamLayer(candidates.map(candidate => ({...candidate})),
    10, "balanced", null, false);
  const metrics = worker.createPerformanceMetrics();
  const selected = worker.selectBeamLayer(candidates, 10, "balanced", metrics, true);
  const classes = entries => new Set(entries.map(candidate =>
    worker.beamFeatureClass(candidate, 10))).size;
  assert.equal(classes(legacy), 1);
  assert.ok(classes(selected) >= 3);
  assert.ok(selected.some(candidate => candidate.exactSignature.startsWith("feature-")));
  assert.ok(metrics.beamFeatureCells >= rareFeatures.length + 1);
  assert.equal(metrics.beamFeatureSelections, 3);
  assert.equal(metrics.beamBandSelections, 7);
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

test("ordering productivity gate periodically resamples an unproductive signal", () => {
  const worker = loadWorker();
  const gate = worker.createOrderingProductivityGate(2, 3);

  assert.equal(gate.shouldEvaluate(), true);
  gate.observe(false);
  assert.equal(gate.shouldEvaluate(), true);
  gate.observe(false);
  assert.deepEqual({...gate.snapshot()}, {evaluated: 0, productive: 0, cooldownRemaining: 3});
  assert.equal(gate.shouldEvaluate(), false);
  assert.equal(gate.shouldEvaluate(), false);
  assert.equal(gate.shouldEvaluate(), false);
  assert.equal(gate.shouldEvaluate(), true);
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

test("bounded push DFS reports liveness by elapsed time", () => {
  const messages = [];
  const worker = loadWorker(message => messages.push(message));
  worker.search({
    algorithm: "bounded-push-dfs",
    state: stateFromRows([
      "OOOOOOOO",
      "O R X SO",
      "OOOOOOOO",
    ]),
    maxVisited: 2,
    progressInterval: 5000,
    progressIntervalMs: -1,
    forcedMacros: false,
  });

  assert.ok(messages.some(message => message.type === "progress" && message.visited < 5000));
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
  assert.equal(typeof progress.performance.strategicOrderingEvaluations, "number");
  assert.equal(typeof progress.performance.strategicOrderingSkips, "number");
  assert.equal(typeof progress.performance.strategicOrderingChanges, "number");
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
  assert.ok(done.performance.graphNodes > 0);
  assert.ok(done.performance.reachabilityCalls > 0);
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
    const after = signature(next.boxes);
    if (after !== before) {
      const reachableBefore = worker.reachablePaths(state, board);
      const retained = worker.pushNeighbors(
        state,
        board,
        reachableBefore,
        {lockProven: true},
      ).some(candidate => signature(candidate.boxes) === after);
      assert.equal(retained, true, `proven commitment must retain known push ${pushes + 1}`);
      pushes++;
      assert.ok(pushes + worker.heuristic(next.boxes, board) <= 252);
    }
    state = next;
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
