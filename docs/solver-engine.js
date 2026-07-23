const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};
const DIRECTION_ENTRIES = Object.entries(DIRS);
const OPPOSITE_DIRECTION_INDEX = [1, 0, 3, 2];
const OPPOSITE = {Up: "Down", Down: "Up", Left: "Right", Right: "Left"};
const MOVE_CODE = {Up: "U", Down: "D", Left: "L", Right: "R"};
const pkey = (y, x) => `${y},${x}`;
function boxSignatureReference(boxes) {
  return boxes.map(box => box.join(",")).sort().join(";");
}

function packedIdentityFromTokens(tokens, board) {
  const sorted = Uint32Array.from(tokens);
  sorted.sort();
  const shift = BigInt(board.dense.tokenBits);
  let identity = BigInt(tokens.length);
  for (const value of sorted) identity = (identity << shift) | BigInt(value);
  return {
    identity,
    signature: [...sorted].map(value => value.toString(36)).join("."),
  };
}

function denseBoxLayout(boxes, board) {
  const cached = board.denseBoxMemo.get(boxes);
  if (cached) return cached;
  const cells = new Uint32Array(boxes.length);
  const labels = new Uint16Array(boxes.length);
  const tokens = new Uint32Array(boxes.length);
  const indexByCell = new Int32Array(board.dense.keys.length);
  const occupancyBits = new Uint32Array(Math.ceil(board.dense.keys.length / 32));
  indexByCell.fill(-1);
  let valid = true;
  for (let index = 0; index < boxes.length; index++) {
    const [y, x, label] = boxes[index];
    const cell = board.dense.idByKey.get(pkey(y, x));
    const labelId = board.dense.labelIds.get(label);
    if (cell === undefined || labelId === undefined) {
      valid = false;
      continue;
    }
    cells[index] = cell;
    labels[index] = labelId;
    tokens[index] = labelId * board.dense.keys.length + cell;
    indexByCell[cell] = index;
    occupancyBits[cell >>> 5] |= 1 << (cell & 31);
  }
  const packed = valid
    ? packedIdentityFromTokens(tokens, board)
    : {identity: null, signature: boxSignatureReference(boxes)};
  const layout = {cells, labels, tokens, indexByCell, occupancyBits, valid, ...packed};
  board.denseBoxMemo.set(boxes, layout);
  board.metrics.denseLayoutBuilds++;
  board.metrics.occupancyWordsBuilt += occupancyBits.length;
  return layout;
}

function deriveDenseBoxLayout(parentBoxes, boxes, changedIndex, destinationId, board) {
  const parent = denseBoxLayout(parentBoxes, board);
  if (!parent.valid) {
    denseBoxLayout(boxes, board);
    return;
  }
  const cells = parent.cells.slice();
  const labels = parent.labels;
  const tokens = parent.tokens.slice();
  const indexByCell = parent.indexByCell.slice();
  const occupancyBits = parent.occupancyBits.slice();
  const previousId = cells[changedIndex];
  cells[changedIndex] = destinationId;
  tokens[changedIndex] = labels[changedIndex] * board.dense.keys.length + destinationId;
  indexByCell[previousId] = -1;
  indexByCell[destinationId] = changedIndex;
  occupancyBits[previousId >>> 5] &= ~(1 << (previousId & 31));
  occupancyBits[destinationId >>> 5] |= 1 << (destinationId & 31);
  const packed = packedIdentityFromTokens(tokens, board);
  board.denseBoxMemo.set(boxes, {
    cells,
    labels,
    tokens,
    indexByCell,
    occupancyBits,
    valid: true,
    ...packed,
  });
  board.metrics.denseLayoutDerivations++;
  board.metrics.occupancyWordCopies += occupancyBits.length;
  board.metrics.denseIdentityUpdates++;
}

function boxSignature(boxes, board = null) {
  const metrics = board?.metrics;
  if (metrics) metrics.signatureCalls++;
  if (board?.boxSignatureMemo.has(boxes)) {
    metrics.signatureCacheHits++;
    return board.boxSignatureMemo.get(boxes);
  }
  const started = metrics ? now() : 0;
  let signature = null;
  if (board) {
    signature = denseBoxLayout(boxes, board).signature;
  }
  signature ??= boxSignatureReference(boxes);
  if (board) board.boxSignatureMemo.set(boxes, signature);
  if (metrics) {
    metrics.signatureMs += now() - started;
    metrics.signatureCharacters += signature.length;
  }
  return signature;
}

function packedBoxIdentity(boxes, board) {
  const metrics = board.metrics;
  metrics.packedIdentityCalls++;
  if (board.boxIdentityMemo.has(boxes)) {
    metrics.packedIdentityCacheHits++;
    return board.boxIdentityMemo.get(boxes);
  }
  const layout = denseBoxLayout(boxes, board);
  if (!layout.valid) {
    throw new Error("Cannot densely encode a box outside the prepared board.");
  }
  const identity = layout.identity;
  board.boxIdentityMemo.set(boxes, identity);
  metrics.packedIdentityValues += layout.tokens.length;
  return identity;
}

function denseStateIdentity(state, board, robotId) {
  if (robotId === undefined || robotId < 0) return exactPushKey(state, board);
  return (packedBoxIdentity(state.boxes, board) << BigInt(board.dense.cellBits)) |
    BigInt(robotId);
}

function exactPushIdentity(state, board) {
  const robotId = board.dense.idByKey.get(pkey(state.robot[0], state.robot[1]));
  return denseStateIdentity(state, board, robotId);
}

function pushIdentity(state, reachable) {
  if (reachable.board && reachable.regionId !== undefined) {
    return denseStateIdentity(state, reachable.board, reachable.regionId);
  }
  return pushKey(state, reachable);
}

function exactPushKey(state, board = null) {
  const robotId = board?.dense.idByKey.get(pkey(state.robot[0], state.robot[1]));
  const robot = robotId === undefined ? state.robot.join(",") : robotId.toString(36);
  return `${robot}|${boxSignature(state.boxes, board)}`;
}
const HEURISTIC_MEMO_LIMIT = 20000;
const DEADLOCK_MEMO_LIMIT = 10000;
const PATTERN_DEADLOCK_MEMO_LIMIT = 10000;
const ROOM_PATTERN_MAX_STATES = 12000;
const PAIR_CONFLICT_MAX_STATES = 4000;
const PAIR_CONFLICT_DISTANCE_LIMIT = 18;
const INCREMENTAL_ASSIGNMENT_CROSSOVER = 3;
function incrementalAssignmentCrossover() {
  return INCREMENTAL_ASSIGNMENT_CROSSOVER;
}
const COMMITMENT_MEMO_LIMIT = 10000;
const SUPPORT_DEPENDENCY_MEMO_LIMIT = 10000;
const LOCAL_ROOM_MEMO_LIMIT = 5000;
const LOCAL_CORRAL_MEMO_LIMIT = 5000;
const DOORWAY_FLOW_MEMO_LIMIT = 10000;
const GOAL_COMMITMENT = Object.freeze({
  TEMPORARY: "temporary",
  CONDITIONAL: "conditional",
  PROVEN: "proven",
});
let activePerformance = null;

const now = () => globalThis.performance?.now?.() ?? Date.now();

function currentHeapSample() {
  let injected = null;
  try {
    injected = globalThis.__sokomindMemoryUsage?.();
  } catch (_error) {
    injected = null;
  }
  if (Number.isFinite(injected) && injected >= 0) {
    return {bytes: Math.round(injected), source: "injected-runtime"};
  }
  const browserHeap = globalThis.performance?.memory?.usedJSHeapSize;
  return Number.isFinite(browserHeap) && browserHeap >= 0
    ? {bytes: Math.round(browserHeap), source: "browser-performance-memory"}
    : null;
}

function samplePerformanceMemory(metrics) {
  const sample = currentHeapSample();
  if (sample === null) return;
  const heap = sample.bytes;
  if (metrics._heapStartBytes === null) metrics._heapStartBytes = heap;
  metrics._heapSource = sample.source;
  metrics.heapSupported = true;
  metrics.heapUsedBytes = heap;
  metrics.heapPeakBytes = Math.max(metrics.heapPeakBytes || 0, heap);
  metrics.heapDeltaBytes = heap - metrics._heapStartBytes;
  metrics.heapSamples++;
}

function createPerformanceMetrics() {
  const metrics = {
    _startedAt: now(),
    _heapStartBytes: null,
    _heapSource: null,
    schemaVersion: 2,
    totalMs: 0,
    heapSupported: false,
    heapUsedBytes: null,
    heapPeakBytes: null,
    heapDeltaBytes: null,
    heapSamples: 0,
    parseMs: 0,
    graphCompileMs: 0,
    graphNodes: 0,
    graphEdges: 0,
    denseCells: 0,
    denseBuildMs: 0,
    denseLayoutBuilds: 0,
    denseLayoutDerivations: 0,
    denseIdentityUpdates: 0,
    occupancyWordsBuilt: 0,
    occupancyWordCopies: 0,
    signatureCalls: 0,
    signatureCacheHits: 0,
    signatureCharacters: 0,
    signatureMs: 0,
    packedIdentityCalls: 0,
    packedIdentityCacheHits: 0,
    packedIdentityValues: 0,
    preparedBoardReuses: 0,
    preparedBoardFallbacks: 0,
    preparedBoardHydrateMs: 0,
    preparedSeedBytes: 0,
    preparedPlayerDistanceTables: 0,
    heuristicCalls: 0,
    heuristicCacheHits: 0,
    heuristicMs: 0,
    commitmentCalls: 0,
    commitmentCacheHits: 0,
    commitmentBoxLocks: 0,
    commitmentMs: 0,
    strategicOrderingEvaluations: 0,
    strategicOrderingSkips: 0,
    strategicOrderingChanges: 0,
    supportDependencyCalls: 0,
    supportDependencyCacheHits: 0,
    supportDependencyOptions: 0,
    supportDependencyBlockers: 0,
    supportDependencyMs: 0,
    localRoomCalls: 0,
    localRoomCacheHits: 0,
    localRoomStates: 0,
    reversePackingBuilds: 0,
    reversePackingStates: 0,
    reversePackingHits: 0,
    roomPatternBuilds: 0,
    roomPatternStates: 0,
    roomPatternHits: 0,
    roomPatternBoost: 0,
    pairConflictBuilds: 0,
    pairConflictStates: 0,
    pairConflictCandidates: 0,
    pairConflictHits: 0,
    pairConflictBoost: 0,
    beamFeatureCells: 0,
    beamFeatureSelections: 0,
    beamBandSelections: 0,
    localRoomMs: 0,
    localCorralCalls: 0,
    localCorralCacheHits: 0,
    localCorralStates: 0,
    localCorralMs: 0,
    doorwayFlowCalls: 0,
    doorwayFlowCacheHits: 0,
    doorwayFlowMs: 0,
    assignmentCalls: 0,
    incrementalAssignmentCalls: 0,
    incrementalAssignmentFallbacks: 0,
    incrementalAssignmentRowsReused: 0,
    pushDistanceCalls: 0,
    pushDistanceCacheHits: 0,
    pushDistanceMs: 0,
    goalTableBuilds: 0,
    goalTableStates: 0,
    goalTableHits: 0,
    goalTableMs: 0,
    reachabilityCalls: 0,
    reachabilityCells: 0,
    reachabilityMs: 0,
    pushNeighborCalls: 0,
    pushCandidates: 0,
    pushesRetained: 0,
    staticDeadPrunes: 0,
    dynamicDeadPrunes: 0,
    patternDeadlockCalls: 0,
    patternDeadlockCacheHits: 0,
    patternDeadlockStates: 0,
    patternDeadlockPrunes: 0,
  };
  samplePerformanceMemory(metrics);
  return metrics;
}

function performanceSnapshot(metrics) {
  samplePerformanceMemory(metrics);
  const rounded = {
    ...metrics,
    totalMs: metrics._startedAt === null ? metrics.totalMs : now() - metrics._startedAt,
  };
  delete rounded._startedAt;
  delete rounded._heapStartBytes;
  delete rounded._heapSource;
  rounded.memory = {
    supported: metrics.heapSupported,
    source: metrics._heapSource,
    usedBytes: metrics.heapUsedBytes,
    peakBytes: metrics.heapPeakBytes,
    deltaBytes: metrics.heapDeltaBytes,
    samples: metrics.heapSamples,
    gcControlled: false,
  };
  for (const key of ["totalMs", "parseMs", "graphCompileMs", "denseBuildMs",
    "preparedBoardHydrateMs", "signatureMs", "heuristicMs", "commitmentMs",
    "supportDependencyMs", "localRoomMs", "localCorralMs", "doorwayFlowMs",
    "pushDistanceMs", "goalTableMs", "reachabilityMs"]) {
    rounded[key] = Math.round((rounded[key] || 0) * 1000) / 1000;
  }
  return rounded;
}

function memoizeBounded(memo, key, value, limit = HEURISTIC_MEMO_LIMIT) {
  if (memo.size >= limit) memo.delete(memo.keys().next().value);
  memo.set(key, value);
  return value;
}

class BoundedDepthMap {
  constructor(limit) {
    this.limit = limit;
    this.values = new Map();
    this.evictions = 0;
  }
  get(key) { return this.values.get(key); }
  has(key) { return this.values.has(key); }
  set(key, value) {
    if (this.values.has(key)) this.values.delete(key);
    this.values.set(key, value);
    while (this.values.size > this.limit) {
      this.values.delete(this.values.keys().next().value);
      this.evictions++;
    }
  }
  get size() { return this.values.size; }
}

function createOrderingProductivityGate(warmup = 64, cooldown = 512) {
  const sampleSize = Math.max(1, Math.floor(warmup) || 1);
  const cooldownSize = Math.max(1, Math.floor(cooldown) || 1);
  let evaluated = 0, productive = 0, cooldownRemaining = 0;
  return {
    shouldEvaluate() {
      if (cooldownRemaining <= 0) return true;
      cooldownRemaining--;
      return false;
    },
    observe(changedOrdering) {
      evaluated++;
      if (changedOrdering) productive++;
      if (evaluated < sampleSize) return;
      if (!productive) cooldownRemaining = cooldownSize;
      evaluated = 0;
      productive = 0;
    },
    snapshot: () => ({evaluated, productive, cooldownRemaining}),
  };
}

class Heap {
  constructor() { this.items = []; }
  push(item) {
    const a = this.items; a.push(item); let i = a.length - 1;
    while (i) { const p = (i - 1) >> 1; if (a[p][0] <= item[0]) break; a[i] = a[p]; i = p; }
    a[i] = item;
  }
  pop() {
    const a = this.items, root = a[0], last = a.pop();
    if (a.length) {
      let i = 0;
      while (true) {
        let c = i * 2 + 1; if (c >= a.length) break;
        if (c + 1 < a.length && a[c + 1][0] < a[c][0]) c++;
        if (a[c][0] >= last[0]) break; a[i] = a[c]; i = c;
      }
      a[i] = last;
    }
    return root;
  }
  retainBest(limit) {
    if (this.items.length <= limit) return;
    const kept = this.items
      .sort((left, right) => left[0] - right[0] || left[1] - right[1])
      .slice(0, limit);
    this.items = [];
    kept.forEach(item => this.push(item));
  }
  get length() { return this.items.length; }
}

function floorNeighbors(position, floor) {
  const [y, x] = position.split(",").map(Number);
  return Object.values(DIRS).map(([dy, dx]) => pkey(y + dy, x + dx))
    .filter(next => floor.has(next));
}

function floorComponents(floor, blocked = null) {
  const remaining = new Set(floor);
  if (blocked) remaining.delete(blocked);
  const components = [];
  while (remaining.size) {
    const start = remaining.values().next().value;
    const component = new Set([start]), queue = [start];
    remaining.delete(start);
    for (let head = 0; head < queue.length; head++) {
      for (const next of floorNeighbors(queue[head], floor)) {
        if (next === blocked || !remaining.has(next)) continue;
        remaining.delete(next);
        component.add(next);
        queue.push(next);
      }
    }
    components.push(component);
  }
  return components;
}

function articulationPoints(floor) {
  const discovered = new Map(), low = new Map(), parent = new Map(), result = new Set();
  let time = 0;
  const visit = position => {
    discovered.set(position, ++time);
    low.set(position, time);
    let children = 0;
    for (const next of floorNeighbors(position, floor)) {
      if (!discovered.has(next)) {
        parent.set(next, position);
        children++;
        visit(next);
        low.set(position, Math.min(low.get(position), low.get(next)));
        if (!parent.has(position) && children > 1) result.add(position);
        if (parent.has(position) && low.get(next) >= discovered.get(position)) result.add(position);
      } else if (next !== parent.get(position)) {
        low.set(position, Math.min(low.get(position), discovered.get(next)));
      }
    }
  };
  for (const position of floor) if (!discovered.has(position)) visit(position);
  return result;
}

function analyzeTopology(floor, goals) {
  const articulations = articulationPoints(floor);
  const tunnels = new Set([...floor].filter(position => {
    const neighbors = floorNeighbors(position, floor);
    if (neighbors.length !== 2) return false;
    const coordinates = neighbors.map(next => next.split(",").map(Number));
    return coordinates[0][0] === coordinates[1][0] || coordinates[0][1] === coordinates[1][1];
  }));
  const candidates = [];
  for (const gate of articulations) {
    const components = floorComponents(floor, gate).sort((left, right) => right.size - left.size);
    for (const cells of components) {
      if (cells.size < 2 || cells.size > floor.size * 0.72) continue;
      const roomGoals = [...goals.keys()].filter(goal => cells.has(goal));
      if (!roomGoals.length) continue;
      const depths = new Map(), queue = [];
      for (const neighbor of floorNeighbors(gate, floor)) {
        if (!cells.has(neighbor)) continue;
        depths.set(neighbor, 1);
        queue.push(neighbor);
      }
      for (let head = 0; head < queue.length; head++) {
        const position = queue[head], distance = depths.get(position);
        for (const next of floorNeighbors(position, floor)) {
          if (!cells.has(next) || depths.has(next)) continue;
          depths.set(next, distance + 1);
          queue.push(next);
        }
      }
      const traffic = new Map([...cells].map(cell => [cell, 0]));
      for (const goal of roomGoals) {
        const goalDistances = new Map([[goal, 0]]), goalQueue = [goal];
        for (let head = 0; head < goalQueue.length; head++) {
          const position = goalQueue[head], distance = goalDistances.get(position);
          for (const next of floorNeighbors(position, floor)) {
            if (!cells.has(next) || goalDistances.has(next)) continue;
            goalDistances.set(next, distance + 1);
            goalQueue.push(next);
          }
        }
        const goalDepth = depths.get(goal);
        for (const cell of cells) {
          if ((depths.get(cell) ?? Infinity) + (goalDistances.get(cell) ?? Infinity) === goalDepth) {
            traffic.set(cell, traffic.get(cell) + 1);
          }
        }
      }
      const dependencies = [];
      for (const blocker of roomGoals) {
        const reducedFloor = new Set(floor);
        reducedFloor.delete(blocker);
        for (const target of roomGoals) {
          if (target === blocker) continue;
          const normallyReachable = reversePushDistances(floor, target).has(gate);
          const blockedReachable = reversePushDistances(reducedFloor, target).has(gate);
          if (normallyReachable && !blockedReachable) dependencies.push([blocker, target]);
        }
      }
      const approach = new Map(), approachQueue = [];
      for (const neighbor of floorNeighbors(gate, floor)) {
        if (cells.has(neighbor)) continue;
        approach.set(neighbor, 1);
        approachQueue.push(neighbor);
      }
      for (let head = 0; head < approachQueue.length; head++) {
        const position = approachQueue[head], distance = approach.get(position);
        if (distance >= 3) continue;
        for (const next of floorNeighbors(position, floor)) {
          if (next === gate || cells.has(next) || approach.has(next)) continue;
          approach.set(next, distance + 1);
          approachQueue.push(next);
        }
      }
      const [gateY, gateX] = gate.split(",").map(Number);
      const doorwayLanes = floorNeighbors(gate, floor)
        .filter(inside => cells.has(inside))
        .map(inside => {
          const [insideY, insideX] = inside.split(",").map(Number);
          const dy = insideY - gateY, dx = insideX - gateX;
          const outside = pkey(gateY - dy, gateX - dx);
          const importSupport = pkey(gateY - 2 * dy, gateX - 2 * dx);
          const exportSupport = pkey(gateY + 2 * dy, gateX + 2 * dx);
          return {
            inside,
            outside,
            importSupport,
            exportSupport,
            importPossible: floor.has(outside) && floor.has(importSupport),
            exportPossible: floor.has(outside) && floor.has(exportSupport),
          };
        });
      const interiorStaging = new Set([...cells].filter(cell => (depths.get(cell) || 0) <= 2));
      const exteriorStaging = new Set([...approach]
        .filter(([, distance]) => distance <= 2)
        .map(([position]) => position));
      candidates.push({
        gate,
        cells,
        goals: roomGoals,
        depths,
        traffic,
        dependencies,
        approach,
        doorwayLanes,
        interiorStaging,
        exteriorStaging,
      });
    }
  }
  candidates.sort((left, right) => right.cells.size - left.cells.size);
  const rooms = [];
  for (const candidate of candidates) {
    if (rooms.some(room => [...candidate.cells].every(cell => room.cells.has(cell)))) continue;
    rooms.push(candidate);
  }
  return {articulations, rooms, tunnels};
}

const PREPARED_BOARD_SCHEMA = 3;
const boardContentKey = rows => rows.join("\n");

function estimatePreparedBoardBytes(board) {
  const stringBytes = values => [...values].reduce((sum, value) => sum + 2 * value.length, 0);
  const nestedMapEntries = map => [...map.values()].reduce(
    (sum, value) => sum + (value?.size || 0),
    0,
  );
  return (
    stringBytes(board.rows) +
    stringBytes(board.floor) +
    stringBytes(board.walls) +
    stringBytes(board.goals.keys()) +
    board.dense.y.byteLength +
    board.dense.x.byteLength +
    board.dense.neighbors.byteLength +
    board.singleBoxGraph.nodes.size * 32 +
    board.metrics.graphEdges * 16 +
    nestedMapEntries(board.pushDistances) * 12 +
    nestedMapEntries(board.goalPushTables.byGoal) * 12 +
    nestedMapEntries(board.playerPushDistances) * 12
  );
}

function createPreparedBoardSeed(board) {
  const estimatedBytes = estimatePreparedBoardBytes(board);
  board.metrics.preparedSeedBytes = estimatedBytes;
  board.metrics.preparedPlayerDistanceTables = board.playerPushDistances.size;
  return {
    schemaVersion: PREPARED_BOARD_SCHEMA,
    boardContentKey: boardContentKey(board.rows),
    floor: board.floor,
    walls: board.walls,
    goals: board.goals,
    goalsByLabel: board.goalsByLabel,
    pushDistances: board.pushDistances,
    goalPressure: board.goalPressure,
    topology: board.topology,
    singleBoxGraph: board.singleBoxGraph,
    goalPushTables: board.goalPushTables,
    playerPushDistances: board.playerPushDistances,
    dense: board.dense,
    goalRoomPackingTables: board.goalRoomPackingTables,
    roomPatternTables: board.roomPatternTables,
    pairConflictTables: board.pairConflictTables,
    graphNodes: board.metrics.graphNodes,
    graphEdges: board.metrics.graphEdges,
    estimatedBytes,
  };
}

function preparedBoardMatches(data, seed) {
  return seed?.schemaVersion === PREPARED_BOARD_SCHEMA &&
    seed.boardContentKey === boardContentKey(data.rows) &&
    typeof seed.floor?.has === "function" && typeof seed.goals?.get === "function" &&
    typeof seed.singleBoxGraph?.nodes?.get === "function" &&
    typeof seed.goalPushTables?.byGoal?.get === "function" &&
    typeof seed.dense?.idByKey?.get === "function";
}

function hydratePreparedBoard(data, seed, metrics) {
  const started = now();
  metrics.preparedBoardReuses++;
  metrics.graphNodes = seed.graphNodes;
  metrics.graphEdges = seed.graphEdges;
  metrics.denseCells = seed.dense.keys.length;
  metrics.preparedSeedBytes = seed.estimatedBytes || 0;
  metrics.preparedPlayerDistanceTables = seed.playerPushDistances?.size || 0;
  metrics.goalTableBuilds = seed.goalPushTables.byGoal.size;
  metrics.goalTableStates = [...seed.goalPushTables.byGoal.values()]
    .reduce((sum, distances) => sum + distances.size, 0);
  const board = {
    rows: data.rows,
    floor: seed.floor,
    walls: seed.walls,
    goals: seed.goals,
    goalsByLabel: seed.goalsByLabel,
    pushDistances: seed.pushDistances,
    goalPressure: seed.goalPressure,
    topology: seed.topology,
    singleBoxGraph: seed.singleBoxGraph,
    goalPushTables: seed.goalPushTables,
    dense: seed.dense,
    heuristicMemo: new Map(),
    assignmentMemo: new WeakMap(),
    assignmentParentMemo: new WeakMap(),
    playerPushDistances: new Map(seed.playerPushDistances || []),
    deadlockMemo: new Map(),
    patternDeadlockMemo: new Map(),
    patternWindowMemo: new Map(),
    commitmentMemo: new Map(),
    stateCommitmentMemo: new Map(),
    commitmentPushDistances: new Map(),
    supportDependencyMemo: new Map(),
    localRoomMemo: new Map(),
    goalRoomPackingTables: new Map(seed.goalRoomPackingTables || []),
    roomPatternTables: new Map(seed.roomPatternTables || []),
    pairConflictTables: new Map(seed.pairConflictTables || []),
    shortestCorridorMemo: new Map(),
    localCorralMemo: new Map(),
    doorwayFlowMemo: new Map(),
    denseBoxMemo: new WeakMap(),
    boxSignatureMemo: new WeakMap(),
    boxIdentityMemo: new WeakMap(),
    metrics,
  };
  metrics.preparedBoardHydrateMs += now() - started;
  return board;
}

function validatePuzzleRows(rows) {
  if (!Array.isArray(rows) || !rows.length) throw new Error("Puzzle is empty.");
  if (rows.some(row => typeof row !== "string")) {
    throw new Error("Every puzzle row must be a string.");
  }
  const reserved = new Set(["O", "R", "S", "X"]);
  const boxCounts = new Map(), goalCounts = new Map();
  let robots = 0;
  rows.forEach((row, y) => [...row].forEach((cell, x) => {
    const uppercase = cell >= "A" && cell <= "Z";
    const lowercase = cell >= "a" && cell <= "z";
    const dedicatedBox = uppercase && !reserved.has(cell);
    const dedicatedGoal = lowercase && !reserved.has(cell.toUpperCase());
    if (!(cell === " " || reserved.has(cell) || dedicatedBox || dedicatedGoal)) {
      throw new Error(`Unsupported symbol ${JSON.stringify(cell)} at row ${y + 1}, column ${x + 1}.`);
    }
    if (cell === "R") robots++;
    if (cell === "X" || dedicatedBox) {
      boxCounts.set(cell, (boxCounts.get(cell) || 0) + 1);
    }
    if (cell === "S") goalCounts.set("X", (goalCounts.get("X") || 0) + 1);
    if (dedicatedGoal) {
      const label = cell.toUpperCase();
      goalCounts.set(label, (goalCounts.get(label) || 0) + 1);
    }
  }));
  if (robots !== 1) {
    throw new Error(`Puzzle must contain exactly one robot; found ${robots}.`);
  }
  const labels = new Set([...boxCounts.keys(), ...goalCounts.keys()]);
  for (const label of [...labels].sort()) {
    const boxes = boxCounts.get(label) || 0, goals = goalCounts.get(label) || 0;
    if (boxes === goals) continue;
    if (label === "X") {
      throw new Error(`Generic boxes/goals mismatch: ${boxes} box(es), ${goals} goal(s).`);
    }
    throw new Error(
      `Dedicated box ${JSON.stringify(label)} has ${boxes} box(es) but ${goals} goal(s).`,
    );
  }
  return true;
}

function parse(data) {
  const parseStarted = now();
  const metrics = activePerformance || createPerformanceMetrics();
  if (preparedBoardMatches(data, data.preparedBoard)) {
    const board = hydratePreparedBoard(data, data.preparedBoard, metrics);
    metrics.parseMs += now() - parseStarted;
    return board;
  }
  if (data.preparedBoard) metrics.preparedBoardFallbacks++;
  const floor = new Set(), walls = new Set(), goals = new Map(), goalsByLabel = new Map();
  data.rows.forEach((row, y) => [...row].forEach((ch, x) => {
    const p = pkey(y, x);
    if (ch === "O") walls.add(p); else floor.add(p);
    const label = ch === "S" ? "X" : /[a-z]/.test(ch) ? ch.toUpperCase() : null;
    if (label) {
      goals.set(p, label);
      if (!goalsByLabel.has(label)) goalsByLabel.set(label, []);
      goalsByLabel.get(label).push(p);
    }
  }));
  const pushDistances = new Map([...goals.keys()].map(goal => [goal, reversePushDistances(floor, goal)]));
  const goalPressure = new Map([...goals.keys()].map(goal => [
    goal,
    floor.size / Math.max(1, pushDistances.get(goal).size),
  ]));
  const dense = compileDenseBoard(floor, goals, metrics);
  const singleBoxGraph = compileSingleBoxPushGraph(floor, metrics);
  const goalPushTables = compileGoalPushTables(singleBoxGraph, goals, metrics);
  const topology = analyzeTopology(floor, goals);
  metrics.parseMs += now() - parseStarted;
  return {
    rows: data.rows, floor, walls, goals, goalsByLabel, pushDistances, goalPressure,
    topology, heuristicMemo: new Map(), assignmentMemo: new WeakMap(),
    assignmentParentMemo: new WeakMap(), playerPushDistances: new Map(),
    deadlockMemo: new Map(), commitmentMemo: new Map(), stateCommitmentMemo: new Map(),
    patternDeadlockMemo: new Map(),
    patternWindowMemo: new Map(),
    commitmentPushDistances: new Map(),
    supportDependencyMemo: new Map(),
    localRoomMemo: new Map(),
    goalRoomPackingTables: new Map(),
    roomPatternTables: new Map(),
    pairConflictTables: new Map(),
    shortestCorridorMemo: new Map(),
    localCorralMemo: new Map(),
    doorwayFlowMemo: new Map(),
    denseBoxMemo: new WeakMap(),
    boxSignatureMemo: new WeakMap(),
    boxIdentityMemo: new WeakMap(),
    singleBoxGraph, goalPushTables, dense, metrics,
  };
}

function compileDenseBoard(floor, goals, metrics) {
  const started = now();
  const keys = [...floor];
  const idByKey = new Map(keys.map((position, id) => [position, id]));
  const y = new Int16Array(keys.length), x = new Int16Array(keys.length);
  const neighbors = new Int32Array(keys.length * DIRECTION_ENTRIES.length);
  neighbors.fill(-1);
  keys.forEach((position, id) => {
    const [cellY, cellX] = position.split(",").map(Number);
    y[id] = cellY;
    x[id] = cellX;
    DIRECTION_ENTRIES.forEach(([, [dy, dx]], direction) => {
      neighbors[id * DIRECTION_ENTRIES.length + direction] =
        idByKey.get(pkey(cellY + dy, cellX + dx)) ?? -1;
    });
  });
  metrics.denseCells = keys.length;
  metrics.denseBuildMs += now() - started;
  const labelIds = new Map([...new Set(goals.values())].sort()
    .map((label, index) => [label, index]));
  const cellBits = Math.max(1, Math.ceil(Math.log2(Math.max(2, keys.length))));
  const tokenCount = Math.max(2, keys.length * Math.max(1, labelIds.size));
  const tokenBits = Math.max(1, Math.ceil(Math.log2(tokenCount)));
  return {keys, idByKey, y, x, neighbors, labelIds, cellBits, tokenBits};
}
function reversePushDistances(floor, goalKey) {
  const [gy, gx] = goalKey.split(",").map(Number);
  const distances = new Map([[goalKey, 0]]), queue = [[gy, gx]];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head], distance = distances.get(pkey(y, x));
    for (const [dy, dx] of Object.values(DIRS)) {
      const previous = pkey(y - dy, x - dx);
      const support = pkey(y - 2 * dy, x - 2 * dx);
      if (!floor.has(previous) || !floor.has(support) || distances.has(previous)) continue;
      distances.set(previous, distance + 1);
      queue.push([y - dy, x - dx]);
    }
  }
  return distances;
}
function minimumAssignment(costs) {
  const size = costs.length;
  if (!size) return {
    cost: 0,
    rowPotential: [0],
    columnPotential: [0],
    matching: [0],
  };
  if (costs.some(row => row.length !== size || row.every(cost => !Number.isFinite(cost)))) {
    return {cost: Infinity};
  }
  const blocked = 1e9;
  const rowPotential = Array(size + 1).fill(0);
  const columnPotential = Array(size + 1).fill(0);
  const matching = Array(size + 1).fill(0);
  const predecessor = Array(size + 1).fill(0);
  for (let row = 1; row <= size; row++) {
    matching[0] = row;
    const minimum = Array(size + 1).fill(blocked);
    const used = Array(size + 1).fill(false);
    let column = 0;
    do {
      used[column] = true;
      const matchedRow = matching[column];
      let delta = blocked, nextColumn = 0;
      for (let candidate = 1; candidate <= size; candidate++) {
        if (used[candidate]) continue;
        const cost = costs[matchedRow - 1][candidate - 1];
        const reduced = (Number.isFinite(cost) ? cost : blocked) -
          rowPotential[matchedRow] - columnPotential[candidate];
        if (reduced < minimum[candidate]) {
          minimum[candidate] = reduced;
          predecessor[candidate] = column;
        }
        if (minimum[candidate] < delta) {
          delta = minimum[candidate];
          nextColumn = candidate;
        }
      }
      if (delta >= blocked) return {cost: Infinity};
      for (let candidate = 0; candidate <= size; candidate++) {
        if (used[candidate]) {
          rowPotential[matching[candidate]] += delta;
          columnPotential[candidate] -= delta;
        } else {
          minimum[candidate] -= delta;
        }
      }
      column = nextColumn;
    } while (matching[column] !== 0);
    do {
      const previous = predecessor[column];
      matching[column] = matching[previous];
      column = previous;
    } while (column !== 0);
  }
  let total = 0;
  for (let column = 1; column <= size; column++) {
    const cost = costs[matching[column] - 1][column - 1];
    if (!Number.isFinite(cost)) return {cost: Infinity};
    total += cost;
  }
  return {cost: total, rowPotential, columnPotential, matching};
}

function repairMinimumAssignment(previous, costs, changedRow) {
  const size = costs.length;
  if (!previous || !Number.isInteger(changedRow) || changedRow < 0 || changedRow >= size ||
      !Number.isFinite(previous.cost) || previous.matching?.length !== size + 1 ||
      costs.some(row => row.length !== size || row.every(cost => !Number.isFinite(cost)))) {
    return null;
  }
  const blocked = 1e9, row = changedRow + 1;
  const rowPotential = [...previous.rowPotential];
  const columnPotential = [...previous.columnPotential];
  const matching = [...previous.matching];
  const freedColumn = matching.findIndex((matchedRow, column) => column > 0 && matchedRow === row);
  if (freedColumn < 1) return null;
  matching[freedColumn] = 0;
  rowPotential[row] = Math.min(...costs[changedRow].map((cost, index) =>
    (Number.isFinite(cost) ? cost : blocked) - columnPotential[index + 1]));

  matching[0] = row;
  const minimum = Array(size + 1).fill(blocked);
  const used = Array(size + 1).fill(false);
  const predecessor = Array(size + 1).fill(0);
  let column = 0;
  do {
    used[column] = true;
    const matchedRow = matching[column];
    let delta = blocked, nextColumn = 0;
    for (let candidate = 1; candidate <= size; candidate++) {
      if (used[candidate]) continue;
      const cost = costs[matchedRow - 1][candidate - 1];
      const reduced = (Number.isFinite(cost) ? cost : blocked) -
        rowPotential[matchedRow] - columnPotential[candidate];
      if (reduced < minimum[candidate]) {
        minimum[candidate] = reduced;
        predecessor[candidate] = column;
      }
      if (minimum[candidate] < delta) {
        delta = minimum[candidate];
        nextColumn = candidate;
      }
    }
    if (delta >= blocked) return {cost: Infinity};
    for (let candidate = 0; candidate <= size; candidate++) {
      if (used[candidate]) {
        rowPotential[matching[candidate]] += delta;
        columnPotential[candidate] -= delta;
      } else {
        minimum[candidate] -= delta;
      }
    }
    column = nextColumn;
  } while (matching[column] !== 0);
  do {
    const previousColumn = predecessor[column];
    matching[column] = matching[previousColumn];
    column = previousColumn;
  } while (column !== 0);

  let total = 0;
  for (let candidate = 1; candidate <= size; candidate++) {
    const cost = costs[matching[candidate] - 1][candidate - 1];
    if (!Number.isFinite(cost)) return {cost: Infinity};
    total += cost;
  }
  return {cost: total, rowPotential, columnPotential, matching};
}

function minimumAssignmentCost(costs) {
  return minimumAssignment(costs).cost;
}

function singleBoxReachable(floor, boxKey, startKey) {
  const reached = new Set([startKey]), queue = [startKey];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head].split(",").map(Number);
    for (const [dy, dx] of Object.values(DIRS)) {
      const next = pkey(y + dy, x + dx);
      if (next === boxKey || !floor.has(next) || reached.has(next)) continue;
      reached.add(next);
      queue.push(next);
    }
  }
  return reached;
}

function compileSingleBoxPushGraph(floor, metrics = createPerformanceMetrics()) {
  const started = now();
  const regionsByBox = new Map();
  for (const boxKey of floor) {
    const components = floorComponents(floor, boxKey);
    const representativeByPosition = new Map();
    const representatives = [];
    for (const component of components) {
      const representative = [...component].sort()[0];
      representatives.push(representative);
      component.forEach(position => representativeByPosition.set(position, representative));
    }
    regionsByBox.set(boxKey, {representativeByPosition, representatives, components});
  }

  const nodes = new Map(), startsByBox = new Map();
  for (const [boxKey, regionData] of regionsByBox) {
    const [y, x] = boxKey.split(",").map(Number);
    const starts = [];
    regionData.components.forEach((component, index) => {
      const representative = regionData.representatives[index];
      const nodeKey = `${boxKey}|${representative}`;
      const transitions = [];
      for (const [dy, dx] of Object.values(DIRS)) {
        const support = pkey(y - dy, x - dx), destination = pkey(y + dy, x + dx);
        if (!component.has(support) || !floor.has(destination)) continue;
        const nextRepresentative = regionsByBox.get(destination)
          .representativeByPosition.get(boxKey);
        if (nextRepresentative === undefined) continue;
        transitions.push({destination, nodeKey: `${destination}|${nextRepresentative}`});
      }
      nodes.set(nodeKey, {boxKey, representative, transitions});
      starts.push(nodeKey);
    });
    startsByBox.set(boxKey, starts);
  }
  metrics.graphCompileMs += now() - started;
  metrics.graphNodes += nodes.size;
  metrics.graphEdges += [...nodes.values()].reduce((sum, node) => sum + node.transitions.length, 0);
  return {nodes, startsByBox};
}

function compileGoalPushTables(singleBoxGraph, goals, metrics = createPerformanceMetrics()) {
  const started = now();
  const predecessors = new Map();
  for (const [nodeKey, node] of singleBoxGraph.nodes) {
    for (const transition of node.transitions) {
      if (!predecessors.has(transition.nodeKey)) predecessors.set(transition.nodeKey, []);
      predecessors.get(transition.nodeKey).push(nodeKey);
    }
  }
  const byGoal = new Map();
  const byLabel = new Map();
  for (const [goal, label] of goals) {
    const nodeDistances = new Map();
    const queue = [];
    for (const nodeKey of singleBoxGraph.startsByBox.get(goal) || []) {
      nodeDistances.set(nodeKey, 0);
      queue.push(nodeKey);
    }
    for (let head = 0; head < queue.length; head++) {
      const nodeKey = queue[head];
      const distance = nodeDistances.get(nodeKey);
      for (const previous of predecessors.get(nodeKey) || []) {
        if (nodeDistances.has(previous)) continue;
        nodeDistances.set(previous, distance + 1);
        queue.push(previous);
      }
    }
    const distances = new Map();
    for (const [box, starts] of singleBoxGraph.startsByBox) {
      let best = Infinity;
      for (const nodeKey of starts) {
        best = Math.min(best, nodeDistances.get(nodeKey) ?? Infinity);
      }
      if (Number.isFinite(best)) distances.set(box, best);
    }
    byGoal.set(goal, distances);
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push({goal, distances});
    metrics.goalTableBuilds++;
    metrics.goalTableStates += nodeDistances.size;
  }
  metrics.goalTableMs += now() - started;
  return {byGoal, byLabel};
}

function compiledGoalPushDistance(board, start, goal) {
  const table = board.goalPushTables.byGoal.get(goal);
  if (!table) return Infinity;
  board.metrics.goalTableHits++;
  return table.get(start) ?? Infinity;
}

function playerAwarePushDistancesReference(floor, startKey) {
  const initialRegions = [], unassigned = new Set(floor);
  unassigned.delete(startKey);
  while (unassigned.size) {
    const representative = unassigned.values().next().value;
    const region = singleBoxReachable(floor, startKey, representative);
    region.forEach(position => unassigned.delete(position));
    initialRegions.push(representative);
  }
  const distances = new Map([[startKey, 0]]), seen = new Set(), queue = [];
  const enqueue = (boxKey, robotKey, distance) => {
    const region = singleBoxReachable(floor, boxKey, robotKey);
    const representative = [...region].sort()[0];
    const signature = `${boxKey}|${representative}`;
    if (seen.has(signature)) return;
    seen.add(signature);
    queue.push({boxKey, region, distance});
  };
  initialRegions.forEach(robotKey => enqueue(startKey, robotKey, 0));
  for (let head = 0; head < queue.length; head++) {
    const {boxKey, region, distance} = queue[head];
    const [y, x] = boxKey.split(",").map(Number);
    for (const [dy, dx] of Object.values(DIRS)) {
      const support = pkey(y - dy, x - dx), destination = pkey(y + dy, x + dx);
      if (!region.has(support) || !floor.has(destination)) continue;
      const nextDistance = distance + 1;
      if (nextDistance < (distances.get(destination) ?? Infinity)) {
        distances.set(destination, nextDistance);
      }
      enqueue(destination, boxKey, nextDistance);
    }
  }
  return distances;
}

function playerAwarePushDistances(board, startKey) {
  const metrics = board.metrics;
  metrics.pushDistanceCalls++;
  if (board.playerPushDistances.has(startKey)) {
    metrics.pushDistanceCacheHits++;
    return board.playerPushDistances.get(startKey);
  }
  const started = now();
  const distances = new Map([[startKey, 0]]), seen = new Set(), queue = [];
  const enqueue = (nodeKey, distance) => {
    if (seen.has(nodeKey)) return;
    seen.add(nodeKey);
    queue.push({nodeKey, distance});
  };
  (board.singleBoxGraph.startsByBox.get(startKey) || []).forEach(nodeKey => enqueue(nodeKey, 0));

  for (let head = 0; head < queue.length; head++) {
    const {nodeKey, distance} = queue[head];
    const node = board.singleBoxGraph.nodes.get(nodeKey);
    for (const transition of node.transitions) {
      const nextDistance = distance + 1;
      if (nextDistance < (distances.get(transition.destination) ?? Infinity)) {
        distances.set(transition.destination, nextDistance);
      }
      enqueue(transition.nodeKey, nextDistance);
    }
  }
  board.playerPushDistances.set(startKey, distances);
  metrics.pushDistanceMs += now() - started;
  return distances;
}

function boxesByLabelWithIndices(boxes) {
  const byLabel = new Map();
  boxes.forEach(([y, x, label], index) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push({y, x, index});
  });
  return byLabel;
}

function cacheFullAssignmentDetail(boxes, board) {
  if (board.assignmentMemo.has(boxes)) return board.assignmentMemo.get(boxes);
  const labels = new Map();
  let total = 0;
  for (const [label, entries] of boxesByLabelWithIndices(boxes)) {
    const targets = board.goalsByLabel.get(label) || [];
    const costs = entries.map(({y, x}) =>
      targets.map(target => compiledGoalPushDistance(board, pkey(y, x), target)));
    board.metrics.assignmentCalls++;
    const assignment = minimumAssignment(costs);
    labels.set(label, {boxIndices: entries.map(entry => entry.index), costs, assignment});
    total += assignment.cost;
  }
  total += interactionHeuristicBoost(
    boxes,
    board,
    new Map([...labels].map(([label, detail]) => [label, detail.assignment.cost])),
  );
  const detail = {labels, cost: total};
  board.assignmentMemo.set(boxes, detail);
  return detail;
}

function heuristic(boxes, board) {
  const metrics = board.metrics;
  metrics.heuristicCalls++;
  const signature = boxSignature(boxes, board);
  if (board.heuristicMemo.has(signature)) {
    metrics.heuristicCacheHits++;
    return board.heuristicMemo.get(signature);
  }
  const started = now();
  const parentHint = board.assignmentParentMemo.get(boxes);
  if (!parentHint) {
    const detail = cacheFullAssignmentDetail(boxes, board);
    metrics.heuristicMs += now() - started;
    return memoizeBounded(board.heuristicMemo, signature, detail.cost);
  }
  const grouped = boxesByLabelWithIndices(boxes);
  const changedEntries = [...grouped.values()].find(entries =>
    entries.some(entry => entry.index === parentHint.changedIndex));
  const parentDetail = changedEntries?.length >= INCREMENTAL_ASSIGNMENT_CROSSOVER
    ? cacheFullAssignmentDetail(parentHint.parentBoxes, board)
    : null;
  let total = 0;
  const assignmentCosts = new Map();
  for (const [label, entries] of grouped) {
    const targets = board.goalsByLabel.get(label) || [];
    const previous = parentDetail?.labels.get(label);
    const boxIndices = entries.map(entry => entry.index);
    const changedRow = parentHint ? boxIndices.indexOf(parentHint.changedIndex) : -1;
    const canReuseRows = changedRow >= 0 && previous &&
      previous.boxIndices.length === boxIndices.length &&
      previous.boxIndices.every((index, row) => index === boxIndices[row]);
    const costs = canReuseRows ? [...previous.costs] : entries.map(({y, x}) =>
      targets.map(target => compiledGoalPushDistance(board, pkey(y, x), target)));
    if (canReuseRows) {
      const {y, x} = entries[changedRow];
      costs[changedRow] = targets.map(target =>
        compiledGoalPushDistance(board, pkey(y, x), target));
    }
    metrics.assignmentCalls++;
    let assignment = canReuseRows
      ? repairMinimumAssignment(previous.assignment, costs, changedRow)
      : null;
    if (canReuseRows) {
      metrics.incrementalAssignmentCalls++;
      metrics.incrementalAssignmentRowsReused += Math.max(0, entries.length - 1);
    }
    if (!assignment) {
      if (canReuseRows) metrics.incrementalAssignmentFallbacks++;
      assignment = minimumAssignment(costs);
    }
    total += assignment.cost;
    assignmentCosts.set(label, assignment.cost);
  }
  total += interactionHeuristicBoost(boxes, board, assignmentCosts);
  metrics.heuristicMs += now() - started;
  return memoizeBounded(board.heuristicMemo, signature, total);
}

function topologyPenalty(boxes, board) {
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  let penalty = 0;
  for (const room of board.topology.rooms) {
    const boxesInside = boxes.filter(([y, x]) => room.cells.has(pkey(y, x)));
    penalty += Math.abs(boxesInside.length - room.goals.length);
    const currentLabels = new Map(), targetLabels = new Map();
    boxesInside.forEach(([, , label]) => currentLabels.set(label, (currentLabels.get(label) || 0) + 1));
    room.goals.forEach(goal => {
      const label = board.goals.get(goal);
      targetLabels.set(label, (targetLabels.get(label) || 0) + 1);
    });
    const labels = new Set([...currentLabels.keys(), ...targetLabels.keys()]);
    let labelFlow = 0;
    labels.forEach(label => {
      labelFlow += Math.abs((currentLabels.get(label) || 0) - (targetLabels.get(label) || 0));
    });
    penalty += 2 * labelFlow;
    for (const [y, x, label] of boxesInside) {
      const position = pkey(y, x);
      if (board.goals.get(position) !== label) penalty += 0.35 * (room.traffic.get(position) || 0);
    }

    const unsolved = room.goals.filter(goal => occupied.get(goal) !== board.goals.get(goal));
    const trafficDemand = unsolved.length + labelFlow;
    for (const [position, distance] of room.approach) {
      if (occupied.has(position)) penalty += 0.75 * trafficDemand * (4 - distance);
    }
    for (const goal of room.goals) {
      if (occupied.get(goal) !== board.goals.get(goal)) continue;
      const depth = room.depths.get(goal) || 0;
      for (const pending of unsolved) {
        const pendingDepth = room.depths.get(pending) || 0;
        if (pendingDepth > depth) penalty += 1 + pendingDepth - depth;
      }
    }
    for (const [blocker, prerequisite] of room.dependencies) {
      const blockerSolved = occupied.get(blocker) === board.goals.get(blocker);
      const prerequisiteSolved = occupied.get(prerequisite) === board.goals.get(prerequisite);
      if (blockerSolved && !prerequisiteSolved) penalty += 4;
    }
    if (occupied.has(room.gate) && unsolved.length) penalty += 2 * unsolved.length;
  }
  return penalty;
}

function roomEvacuationPenalty(boxes, board) {
  const occupied = new Set(boxes.map(([y, x]) => pkey(y, x)));
  let penalty = 0;
  for (const room of board.topology.rooms) {
    const current = new Map(), target = new Map();
    const inside = boxes.filter(([y, x]) => room.cells.has(pkey(y, x)));
    inside.forEach(([, , label]) => current.set(label, (current.get(label) || 0) + 1));
    room.goals.forEach(goal => {
      const label = board.goals.get(goal);
      target.set(label, (target.get(label) || 0) + 1);
    });
    let surplus = 0;
    current.forEach((count, label) => {
      surplus += Math.max(0, count - (target.get(label) || 0));
    });
    if (!surplus) continue;
    penalty += 20 * inside.length + 8 * surplus;
    for (const [position, distance] of room.approach) {
      if (occupied.has(position)) penalty += 3 * (4 - distance);
    }
  }
  return penalty;
}

function typedDoorwayFlow(boxes, board) {
  const metrics = board.metrics;
  metrics.doorwayFlowCalls++;
  const signature = boxSignature(boxes, board);
  if (board.doorwayFlowMemo.has(signature)) {
    metrics.doorwayFlowCacheHits++;
    return board.doorwayFlowMemo.get(signature);
  }
  const started = now();
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  let penalty = 0;
  const rooms = board.topology.rooms.map((room, index) => {
    const current = new Map(), target = new Map();
    boxes.forEach(([y, x, label]) => {
      if (room.cells.has(pkey(y, x))) current.set(label, (current.get(label) || 0) + 1);
    });
    room.goals.forEach(goal => {
      const label = board.goals.get(goal);
      target.set(label, (target.get(label) || 0) + 1);
    });
    const imports = new Map(), exports = new Map();
    for (const label of new Set([...current.keys(), ...target.keys()])) {
      const difference = (target.get(label) || 0) - (current.get(label) || 0);
      if (difference > 0) imports.set(label, difference);
      if (difference < 0) exports.set(label, -difference);
    }
    const importTotal = [...imports.values()].reduce((sum, count) => sum + count, 0);
    const exportTotal = [...exports.values()].reduce((sum, count) => sum + count, 0);
    const interiorCapacity = [...room.interiorStaging].filter(cell => !occupied.has(cell)).length;
    const exteriorCapacity = [...room.exteriorStaging].filter(cell => !occupied.has(cell)).length;
    const importLanes = room.doorwayLanes.filter(lane => lane.importPossible);
    const exportLanes = room.doorwayLanes.filter(lane => lane.exportPossible);
    const readyImportLanes = importLanes.filter(lane =>
      !occupied.has(room.gate) && !occupied.has(lane.inside) && !occupied.has(lane.importSupport));
    const readyExportLanes = exportLanes.filter(lane =>
      !occupied.has(room.gate) && !occupied.has(lane.outside) && !occupied.has(lane.exportSupport));
    const contradictions = [];
    if (importTotal && !importLanes.length) contradictions.push("no-import-lane");
    if (exportTotal && !exportLanes.length) contradictions.push("no-export-lane");
    if (importTotal && !exteriorCapacity) contradictions.push("exterior-staging-full");
    if (exportTotal && !interiorCapacity) contradictions.push("interior-staging-full");
    const gateLabel = occupied.get(room.gate) || null;
    const crossings = importTotal + exportTotal;
    const roomPenalty = crossings +
      (gateLabel && crossings ? 2 * crossings : 0) +
      (importTotal && !readyImportLanes.length ? 2 : 0) +
      (exportTotal && !readyExportLanes.length ? 2 : 0) +
      2 * contradictions.length;
    penalty += roomPenalty;
    return {
      index,
      room,
      imports,
      exports,
      importTotal,
      exportTotal,
      interiorCapacity,
      exteriorCapacity,
      readyImportLanes: readyImportLanes.length,
      readyExportLanes: readyExportLanes.length,
      gateLabel,
      contradictions,
      penalty: roomPenalty,
    };
  });
  const result = {rooms, penalty};
  metrics.doorwayFlowMs += now() - started;
  return memoizeBounded(board.doorwayFlowMemo, signature, result, DOORWAY_FLOW_MEMO_LIMIT);
}

function doorwayFlowDelta(analysis, state, next) {
  const from = next.pushedFrom, to = next.pushedTo;
  const label = state.boxes.find(([y, x]) => pkey(y, x) === from)?.[2] ||
    next.pushClass?.split(":")[0];
  let delta = 0;
  for (const flow of analysis.rooms) {
    const {room} = flow;
    const fromInside = room.cells.has(from), toInside = room.cells.has(to);
    let direction = null;
    if ((to === room.gate && !fromInside) || (from === room.gate && toInside)) {
      direction = "import";
    } else if ((to === room.gate && fromInside) || (from === room.gate && !toInside)) {
      direction = "export";
    }
    if (direction === "import") delta += flow.imports.has(label) ? -1.5 : 1.5;
    if (direction === "export") delta += flow.exports.has(label) ? -1.5 : 1.5;
    if (flow.importTotal) {
      if (room.exteriorStaging.has(to)) delta += 0.25;
      if (room.exteriorStaging.has(from) && flow.imports.has(label)) delta -= 0.25;
    }
    if (flow.exportTotal) {
      if (room.interiorStaging.has(to)) delta += 0.25;
      if (room.interiorStaging.has(from) && flow.exports.has(label)) delta -= 0.25;
    }
  }
  return delta;
}

function roomFlowSignature(boxes, board) {
  return board.topology.rooms.map((room, index) => {
    const labels = boxes
      .filter(([y, x]) => room.cells.has(pkey(y, x)))
      .map(([, , label]) => label)
      .sort()
      .join("");
    return `${index}:${labels}`;
  }).join("|");
}

function roomTransitionEvent(before, after, board) {
  const previous = roomFlowSignature(before, board);
  const current = roomFlowSignature(after, board);
  return previous === current ? null : current;
}

function analyzePuzzleForSearch(data) {
  const board = parse(data);
  const boxes = data.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  const initial = {robot: data.robot, boxes};
  const labels = {};
  boxes.forEach(([, , label]) => { labels[label] = (labels[label] || 0) + 1; });
  const initialHeuristic = heuristic(boxes, board);
  const evacuationPenalty = roomEvacuationPenalty(boxes, board);
  const legalPushes = pushNeighbors(initial, board).length;
  const solvedBoxes = boxes.filter(([y, x, label]) =>
    board.goals.get(pkey(y, x)) === label).length;
  const roomSummaries = board.topology.rooms.map(room => {
    const inside = boxes.filter(([y, x]) => room.cells.has(pkey(y, x)));
    const current = {}, target = {};
    inside.forEach(([, , label]) => { current[label] = (current[label] || 0) + 1; });
    room.goals.forEach(goal => {
      const label = board.goals.get(goal);
      target[label] = (target[label] || 0) + 1;
    });
    const surplus = Object.entries(current).reduce((sum, [label, count]) =>
      sum + Math.max(0, count - (target[label] || 0)), 0);
    return {
      gate: room.gate,
      cells: room.cells.size,
      goals: room.goals.length,
      boxes: inside.length,
      surplus,
      dependencies: room.dependencies.length,
      maxDepth: Math.max(0, ...room.depths.values()),
    };
  });
  const searchScale = boxes.length * board.floor.size;
  const dependencyCount = roomSummaries.reduce((sum, room) => sum + room.dependencies, 0);
  const surplusBoxes = roomSummaries.reduce((sum, room) => sum + room.surplus, 0);
  const reversePortfolio = reverseStartPortfolio(board, boxes);
  const productiveReverseRegions = reversePortfolio.filter(entry => entry.pullOptions > 0);
  for (const [y, x] of boxes) playerAwarePushDistances(board, pkey(y, x));
  const pressure = searchScale * (1 + 0.18 * board.topology.rooms.length) *
    (1 + 0.08 * dependencyCount) * (1 + 0.06 * Math.max(0, legalPushes - 2));
  const difficulty = pressure >= 1800 ? "extreme" : pressure >= 700 ? "complex" :
    pressure >= 180 ? "moderate" : "small";
  const phases = [];
  if (surplusBoxes) phases.push({id: "evacuation", reason: `${surplusBoxes} surplus room box${surplusBoxes === 1 ? "" : "es"}`});
  if (board.topology.rooms.length) phases.push({id: "room-packing", reason: `${board.topology.rooms.length} gated goal room${board.topology.rooms.length === 1 ? "" : "s"}`});
  if (board.topology.tunnels.size) phases.push({id: "tunnel-macros", reason: `${board.topology.tunnels.size} tunnel cells`});
  if (difficulty === "complex" || difficulty === "extreme") {
    phases.push({id: "milestone-reverse", reason: "large canonical push space"});
    phases.push({id: "landmark-bridges", reason: "connect forward phases to reverse layouts"});
  }
  phases.push({id: "exact-proof", reason: "complete fallback after heuristic workers"});
  const recommendations = {
    reverseWorkerLimit: difficulty === "extreme" ? 2 : difficulty === "complex" ? 2 : 3,
    sideVisitedLimit: difficulty === "extreme" ? 100000 : difficulty === "complex" ? 200000 : 250000,
    beamAttempts: difficulty === "small" ? 1 : 2,
    beamWidth: difficulty === "extreme" ? 300 : difficulty === "complex" ? 700 : 1200,
    beamVisited: difficulty === "extreme" ? 110000 : difficulty === "complex" ? 180000 : 250000,
    useEvacuation: surplusBoxes > 0,
    useSequenceMacros: board.topology.tunnels.size > 0 || board.topology.rooms.length > 0,
    useMilestoneReverse: difficulty === "complex" || difficulty === "extreme",
    checkpointLimit: difficulty === "extreme" ? 12 : 8,
  };
  const preparedBoard = createPreparedBoardSeed(board);
  return {
    dimensions: {rows: data.rows.length, columns: Math.max(...data.rows.map(row => row.length))},
    floorCells: board.floor.size,
    boxes: boxes.length,
    goals: board.goals.size,
    labels,
    solvedBoxes,
    initialHeuristic,
    legalPushes,
    articulations: board.topology.articulations.size,
    tunnelCells: board.topology.tunnels.size,
    rooms: roomSummaries,
    surplusBoxes,
    evacuationPenalty,
    dependencyCount,
    reverseStartRegions: reversePortfolio.length,
    productiveReverseStartRegions: productiveReverseRegions.length,
    reverseStartPulls: reversePortfolio.reduce((sum, entry) => sum + entry.pullOptions, 0),
    searchScale,
    pressure: Math.round(pressure),
    difficulty,
    phases,
    recommendations,
    preparedBoard,
    preparedBoardStats: {
      estimatedBytes: preparedBoard.estimatedBytes,
      goalTables: preparedBoard.goalPushTables.byGoal.size,
      playerDistanceTables: preparedBoard.playerPushDistances.size,
      graphNodes: board.metrics.graphNodes,
      graphEdges: board.metrics.graphEdges,
      buildMs: Math.round(board.metrics.parseMs * 1000) / 1000,
    },
  };
}

function stratifiedCheckpoints(checkpoints) {
  const bands = new Map();
  for (const checkpoint of checkpoints) {
    if (!bands.has(checkpoint.checkpointBand)) bands.set(checkpoint.checkpointBand, []);
    bands.get(checkpoint.checkpointBand).push(checkpoint);
  }
  const queues = [...bands.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, items]) => items);
  const result = [];
  for (let depth = 0; queues.some(items => depth < items.length); depth++) {
    for (const items of queues) if (depth < items.length) result.push(items[depth]);
  }
  return result;
}
function targetMapFromBoxes(boxes, board) {
  const targets = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!targets.has(label)) targets.set(label, []);
    targets.get(label).push({distances: playerAwarePushDistances(board, pkey(y, x))});
  });
  targets.memo = new Map();
  targets.board = board;
  return targets;
}
function homeHeuristic(boxes, targetsByLabel) {
  const signature = boxSignature(boxes, targetsByLabel.board);
  if (targetsByLabel.memo.has(signature)) return targetsByLabel.memo.get(signature);
  const byLabel = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push([y, x]);
  });
  let total = 0;
  for (const [label, positions] of byLabel) {
    const targets = targetsByLabel.get(label) || [];
    const costs = positions.map(([y, x]) => targets.map(target => (
      target.distances.get(pkey(y, x)) ?? Infinity
    )));
    total += minimumAssignmentCost(costs);
  }
  return memoizeBounded(targetsByLabel.memo, signature, total);
}

function targetLayoutHeuristic(boxes, targetBoxes, board, memo) {
  const signature = boxSignature(boxes, board);
  if (memo.has(signature)) return memo.get(signature);
  const targetsByLabel = new Map();
  targetBoxes.forEach(([y, x, label]) => {
    if (!targetsByLabel.has(label)) targetsByLabel.set(label, []);
    targetsByLabel.get(label).push(pkey(y, x));
  });
  const byLabel = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push([y, x]);
  });
  let total = 0;
  for (const [label, positions] of byLabel) {
    const targets = targetsByLabel.get(label) || [];
    const costs = positions.map(([y, x]) => {
      const distances = playerAwarePushDistances(board, pkey(y, x));
      return targets.map(target => distances.get(target) ?? Infinity);
    });
    total += minimumAssignmentCost(costs);
  }
  return memoizeBounded(memo, signature, total);
}
function goal(boxes, goals) {
  return boxes.every(([y, x, label]) => goals.get(pkey(y, x)) === label);
}

function staticallyImmovable(position, board) {
  const [y, x] = position.split(",").map(Number);
  return !Object.values(DIRS).some(([dy, dx]) =>
    board.floor.has(pkey(y + dy, x + dx)) &&
    board.floor.has(pkey(y - dy, x - dx)));
}

function commitmentPushDistances(board, fixedPosition, target) {
  const key = `${fixedPosition}|${target}`;
  board.commitmentPushDistances ??= new Map();
  if (board.commitmentPushDistances.has(key)) return board.commitmentPushDistances.get(key);
  const floor = new Set(board.floor);
  floor.delete(fixedPosition);
  const distances = reversePushDistances(floor, target);
  board.commitmentPushDistances.set(key, distances);
  return distances;
}

function residualMatchingSurvives(boxes, board, fixedIndex, fixedPosition) {
  const remaining = boxes.filter((_, index) => index !== fixedIndex);
  const boxesByLabel = new Map(), goalsByLabel = new Map();
  for (const [y, x, label] of remaining) {
    if (!boxesByLabel.has(label)) boxesByLabel.set(label, []);
    boxesByLabel.get(label).push(pkey(y, x));
  }
  for (const [position, label] of board.goals) {
    if (position === fixedPosition) continue;
    if (!goalsByLabel.has(label)) goalsByLabel.set(label, []);
    goalsByLabel.get(label).push(position);
  }
  const labels = new Set([...boxesByLabel.keys(), ...goalsByLabel.keys()]);
  for (const label of labels) {
    const positions = boxesByLabel.get(label) || [];
    const targets = goalsByLabel.get(label) || [];
    if (positions.length !== targets.length) return false;
    const distances = targets.map(target =>
      commitmentPushDistances(board, fixedPosition, target));
    const costs = positions.map(position =>
      distances.map(distance => distance.get(position) ?? Infinity));
    if (!Number.isFinite(minimumAssignmentCost(costs))) return false;
  }
  return true;
}

function blocksPendingRoomWork(position, occupied, board) {
  for (const room of board.topology.rooms) {
    if (!room.cells.has(position) && room.gate !== position) continue;
    const pending = room.goals.filter(goal =>
      goal !== position && occupied.get(goal) !== board.goals.get(goal));
    if (room.gate === position && pending.length) return true;
    if (room.dependencies.some(([blocker, prerequisite]) =>
      blocker === position &&
      occupied.get(prerequisite) !== board.goals.get(prerequisite))) return true;
  }
  return false;
}

function commitmentPositionConflict(position, evidence) {
  const dependencyGraph = evidence?.supportDependency;
  if ((dependencyGraph?.supportDemand?.get(position) || 0) > 0 ||
      (dependencyGraph?.prerequisiteDemand?.get(position) || 0) > 0) return true;
  for (const flow of evidence?.doorway?.rooms || []) {
    const crossings = flow.importTotal + flow.exportTotal;
    if (crossings && position === flow.room.gate) return true;
    if (flow.importTotal && flow.room.exteriorStaging.has(position)) return true;
    if (flow.exportTotal && flow.room.interiorStaging.has(position)) return true;
  }
  for (const analysis of evidence?.localAnalyses || []) {
    if (analysis.status !== "solvable" || !analysis.firstPushes?.size) continue;
    const requiredSources = new Set(
      [...analysis.firstPushes].map(push => push.split(">")[0]),
    );
    if (requiredSources.size === 1 && requiredSources.has(position)) return true;
  }
  return false;
}

function exactRoomCompletionProven(analysis, evidence) {
  if (analysis.kind === "corral" || analysis.roomIndex === undefined) return false;
  const transition = evidence?.transition;
  if (analysis.status === "packed") {
    return !transition ||
      (!analysis.domain.has(transition.pushedFrom) && !analysis.domain.has(transition.pushedTo));
  }
  return analysis.status === "solvable" && analysis.pushes === 1 &&
    transition?.pushes === 1 &&
    analysis.firstPushes.has(`${transition.pushedFrom}>${transition.pushedTo}`);
}

function refineGoalCommitments(base, boxes, board, evidence) {
  const commitments = new Map(base);
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  for (const position of commitments.keys()) {
    if (commitmentPositionConflict(position, evidence)) {
      commitments.set(position, GOAL_COMMITMENT.TEMPORARY);
    }
  }
  for (const analysis of evidence?.localAnalyses || []) {
    if (!exactRoomCompletionProven(analysis, evidence)) continue;
    const room = board.topology.rooms[analysis.roomIndex];
    const flow = evidence?.doorway?.rooms?.find(candidate => candidate.index === analysis.roomIndex);
    if (!room || !flow || flow.importTotal || flow.exportTotal || flow.contradictions.length ||
        flow.gateLabel) continue;
    for (const position of room.cells) {
      const label = occupied.get(position);
      if (label === undefined || board.goals.get(position) !== label ||
          commitmentPositionConflict(position, evidence)) continue;
      if (commitments.get(position) !== GOAL_COMMITMENT.TEMPORARY) {
        commitments.set(position, GOAL_COMMITMENT.PROVEN);
      }
    }
  }
  return commitments;
}

function goalCommitments(boxes, board, evidence = null) {
  const metrics = board.metrics;
  if (metrics) metrics.commitmentCalls++;
  const signature = boxSignature(boxes, board);
  if (board.commitmentMemo.has(signature)) {
    if (metrics) metrics.commitmentCacheHits++;
    const cached = board.commitmentMemo.get(signature);
    return evidence ? refineGoalCommitments(cached, boxes, board, evidence) : cached;
  }
  const started = metrics ? now() : 0;
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  const commitments = new Map();
  boxes.forEach(([y, x, label], index) => {
    const position = pkey(y, x);
    if (board.goals.get(position) !== label) return;
    const safeForRemaining = !blocksPendingRoomWork(position, occupied, board) &&
      residualMatchingSurvives(boxes, board, index, position);
    commitments.set(position, !safeForRemaining
      ? GOAL_COMMITMENT.TEMPORARY
      : staticallyImmovable(position, board)
        ? GOAL_COMMITMENT.PROVEN
        : GOAL_COMMITMENT.CONDITIONAL);
  });
  const result = memoizeBounded(
    board.commitmentMemo,
    signature,
    commitments,
    COMMITMENT_MEMO_LIMIT,
  );
  if (metrics) metrics.commitmentMs += now() - started;
  return evidence ? refineGoalCommitments(result, boxes, board, evidence) : result;
}

function goalPackingBonus(boxes, board, evidence = null) {
  const commitments = goalCommitments(boxes, board, evidence);
  return boxes.reduce((bonus, [y, x]) => {
    const position = pkey(y, x);
    const commitment = commitments.get(position);
    const safetyWeight = commitment === GOAL_COMMITMENT.PROVEN
      ? 1
      : commitment === GOAL_COMMITMENT.CONDITIONAL ? 0.25 : 0;
    return bonus + safetyWeight * (board.goalPressure.get(position) || 0);
  }, 0);
}

function stateCommitmentEvidence(state, board, reachable = reachablePaths(state, board)) {
  return {
    doorway: typedDoorwayFlow(state.boxes, board),
    supportDependency: supportDependencyGraph(state, board, reachable),
    localAnalyses: [
      ...exactLocalRoomAnalyses(state, board, reachable),
      ...exactLocalCorralAnalyses(state, board, reachable),
    ],
  };
}

function stateGoalCommitments(state, board, reachable = reachablePaths(state, board)) {
  const region = reachable.regionId ?? [...reachable.keys()].sort()[0] ?? "sealed";
  const key = `${region}|${boxSignature(state.boxes, board)}`;
  if (board.stateCommitmentMemo.has(key)) return board.stateCommitmentMemo.get(key);
  const commitments = goalCommitments(
    state.boxes,
    board,
    stateCommitmentEvidence(state, board, reachable),
  );
  return memoizeBounded(
    board.stateCommitmentMemo,
    key,
    commitments,
    COMMITMENT_MEMO_LIMIT,
  );
}
function corner(y, x, board, label) {
  if (board.goals.get(pkey(y, x)) === label) return false;
  const wall = (dy, dx) => board.walls.has(pkey(y + dy, x + dx));
  return (wall(-1, 0) && wall(0, -1)) || (wall(-1, 0) && wall(0, 1)) ||
    (wall(1, 0) && wall(0, -1)) || (wall(1, 0) && wall(0, 1));
}
function staticDead(y, x, board, label) {
  if (board.goals.get(pkey(y, x)) === label) return false;
  const distances = playerAwarePushDistances(board, pkey(y, x));
  return !(board.goalsByLabel.get(label) || []).some(goal => distances.has(goal));
}
function creates2x2Deadlock(boxes, board, movedBox) {
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  const [boxY, boxX] = movedBox;
  for (const originY of [boxY - 1, boxY]) {
    for (const originX of [boxX - 1, boxX]) {
      const cells = [
        [originY, originX], [originY + 1, originX],
        [originY, originX + 1], [originY + 1, originX + 1],
      ];
      if (!cells.every(([y, x]) => board.walls.has(pkey(y, x)) || occupied.has(pkey(y, x)))) continue;
      if (cells.some(([y, x]) => {
        const label = occupied.get(pkey(y, x));
        return label && board.goals.get(pkey(y, x)) !== label;
      })) return true;
    }
  }
  return false;
}
function createsFrozenComponentDeadlock(boxes, board, movedBox) {
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  const start = pkey(movedBox[0], movedBox[1]);
  if (!occupied.has(start)) return false;
  const component = new Set([start]), queue = [movedBox];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head];
    for (const [dy, dx] of Object.values(DIRS)) {
      const adjacent = pkey(y + dy, x + dx);
      if (!occupied.has(adjacent) || component.has(adjacent)) continue;
      component.add(adjacent);
      queue.push([y + dy, x + dx]);
    }
  }
  const movable = queue.some(([y, x]) => Object.values(DIRS).some(([dy, dx]) => {
    const destination = pkey(y + dy, x + dx);
    const support = pkey(y - dy, x - dx);
    return board.floor.has(destination) && board.floor.has(support) &&
      !occupied.has(destination) && !occupied.has(support);
  }));
  if (movable) return false;
  return [...component].some(position => board.goals.get(position) !== occupied.get(position));
}

function createsClosedDiagonalDeadlock(boxes, board, movedBox) {
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  const movedKey = pkey(movedBox[0], movedBox[1]);
  if (!occupied.has(movedKey)) return false;
  const limit = board.rows.length + Math.max(...board.rows.map(row => row.length)) + 2;
  const blocked = (y, x) => board.walls.has(pkey(y, x)) || occupied.has(pkey(y, x));

  const scanHalf = (startY, startX, stepY, stepX) => {
    const boxesOnBorder = new Set();
    const boxSides = [];
    let y = startY, x = startX;
    for (let distance = 0; distance < limit; distance++, y += stepY, x += stepX) {
      const center = pkey(y, x);
      if (board.walls.has(center)) {
        return {closed: true, boxes: boxesOnBorder, boxSides, rows: distance};
      }
      if (!board.floor.has(center) || occupied.has(center) || board.goals.has(center)) {
        return {closed: false, boxes: boxesOnBorder, boxSides, rows: distance};
      }
      let rowBoxSide = null;
      for (const [sideOffset, sideX] of [[-1, x - 1], [1, x + 1]]) {
        const side = pkey(y, sideX);
        if (!blocked(y, sideX) || board.goals.has(side)) {
          return {closed: false, boxes: boxesOnBorder, boxSides, rows: distance};
        }
        if (occupied.has(side)) {
          if (rowBoxSide !== null) {
            return {closed: false, boxes: boxesOnBorder, boxSides, rows: distance};
          }
          rowBoxSide = sideOffset;
          boxesOnBorder.add(side);
        }
      }
      if (rowBoxSide === null) {
        return {closed: false, boxes: boxesOnBorder, boxSides, rows: distance};
      }
      boxSides.push(rowBoxSide);
    }
    return {closed: false, boxes: boxesOnBorder, boxSides, rows: limit};
  };

  for (const centerX of [movedBox[1] - 1, movedBox[1] + 1]) {
    for (const slope of [-1, 1]) {
      const up = scanHalf(movedBox[0], centerX, -1, -slope);
      const down = scanHalf(movedBox[0] + 1, centerX + slope, 1, slope);
      if (!up.closed || !down.closed || up.rows + down.rows < 2) continue;
      const participants = new Set([...up.boxes, ...down.boxes]);
      const boxSides = [...up.boxSides].reverse().concat(down.boxSides);
      const outwardFacing = boxSides.length === 2 &&
        boxSides[0] === -slope && boxSides[1] === slope;
      if (outwardFacing && participants.has(movedKey) && participants.size >= 2) return true;
    }
  }
  return false;
}

function createsPatternDatabaseDeadlock(boxes, board, movedBox, maxStates = 64) {
  const metrics = board.metrics;
  metrics.patternDeadlockCalls++;
  const [centerY, centerX] = movedBox;
  const inside = position => {
    const [y, x] = position.split(",").map(Number);
    return Math.abs(y - centerY) <= 4 && Math.abs(x - centerX) <= 4;
  };
  const windowKey = `${centerY},${centerX}`;
  let window = board.patternWindowMemo.get(windowKey);
  if (!window) {
    const floor = new Set([...board.floor].filter(inside));
    const eligible = floor.size <= 24 &&
      ![...floor].some(position => floorNeighbors(position, board.floor).length > 2);
    window = {floor, eligible};
    board.patternWindowMemo.set(windowKey, window);
  }
  if (!window.eligible) return false;
  const localFloor = window.floor;
  const localBoxes = boxes
    .map(([y, x, label]) => [pkey(y, x), label])
    .filter(([position]) => localFloor.has(position));
  if (localBoxes.length < 2 || localBoxes.length > 5) return false;
  if (new Set(localBoxes.map(([, label]) => label)).size < 2) return false;
  const localSignature = localBoxes
    .map(([position, label]) => `${position},${label}`).sort().join(";");
  const cacheKey = `${centerY},${centerX}|${localSignature}`;
  if (board.patternDeadlockMemo.has(cacheKey)) {
    metrics.patternDeadlockCacheHits++;
    return board.patternDeadlockMemo.get(cacheKey);
  }

  const signature = local => local
    .map(([position, label]) => `${position},${label}`).sort().join(";");
  const queue = [localBoxes], seen = new Set([signature(localBoxes)]);
  let head = 0, salvaged = false;
  for (; head < queue.length && seen.size <= maxStates; head++) {
    const current = queue[head];
    if (current.every(([position, label]) => board.goals.get(position) === label)) {
      salvaged = true;
      break;
    }
    const occupied = new Set(current.map(([position]) => position));
    current.forEach(([position, label], boxIndex) => {
      const [y, x] = position.split(",").map(Number);
      for (const [dy, dx] of Object.values(DIRS)) {
        const support = pkey(y - dy, x - dx);
        const destination = pkey(y + dy, x + dx);
        if (!board.floor.has(support) || occupied.has(support) ||
            !board.floor.has(destination) || occupied.has(destination)) continue;
        const next = inside(destination)
          ? current.map((box, index) => index === boxIndex ? [destination, label] : box)
          : current.filter((_, index) => index !== boxIndex);
        const nextSignature = signature(next);
        if (seen.has(nextSignature)) continue;
        seen.add(nextSignature);
        queue.push(next);
      }
    });
  }
  metrics.patternDeadlockStates += Math.min(head, queue.length);
  const cutoff = !salvaged && head < queue.length;
  const deadlocked = !salvaged && !cutoff;
  if (deadlocked) metrics.patternDeadlockPrunes++;
  return memoizeBounded(
    board.patternDeadlockMemo,
    cacheKey,
    deadlocked,
    PATTERN_DEADLOCK_MEMO_LIMIT,
  );
}

function createsDynamicDeadlock(boxes, board, movedBox) {
  const signature = `${boxSignature(boxes, board)}|${movedBox.join(",")}`;
  if (board.deadlockMemo.has(signature)) return board.deadlockMemo.get(signature);
  const deadlocked = DYNAMIC_HARD_PRUNING_RULES.some(
    rule => rule.detect(boxes, board, movedBox),
  );
  return memoizeBounded(board.deadlockMemo, signature, deadlocked, DEADLOCK_MEMO_LIMIT);
}
function neighbors(state, board, pruneDeadlocks = true) {
  const occupied = denseOccupancy(state, board), result = [];
  const robotId = board.dense.idByKey.get(pkey(state.robot[0], state.robot[1]));
  for (let direction = 0; direction < DIRECTION_ENTRIES.length; direction++) {
    const [move, [dy, dx]] = DIRECTION_ENTRIES[direction];
    const [y, x] = state.robot;
    const nextId = board.dense.neighbors[robotId * DIRECTION_ENTRIES.length + direction];
    if (nextId < 0) continue;
    const ny = board.dense.y[nextId], nx = board.dense.x[nextId];
    let boxes = state.boxes;
    if (occupied[nextId] >= 0) {
      const beyondId = board.dense.neighbors[nextId * DIRECTION_ENTRIES.length + direction];
      if (beyondId < 0 || occupied[beyondId] >= 0) continue;
      const by = board.dense.y[beyondId], bx = board.dense.x[beyondId];
      const index = occupied[nextId], label = boxes[index][2];
      boxes = boxes.slice();
      boxes[index] = [by, bx, label];
      if (pruneDeadlocks && staticDead(by, bx, board, label)) continue;
      deriveDenseBoxLayout(state.boxes, boxes, index, beyondId, board);
      if (pruneDeadlocks && createsDynamicDeadlock(boxes, board, [by, bx])) continue;
    }
    result.push({robot: [ny, nx], boxes, move});
  }
  return result;
}

function reachablePathsReference(state, board) {
  const occupied = new Set(state.boxes.map(b => pkey(b[0], b[1])));
  const start = pkey(state.robot[0], state.robot[1]);
  const parents = new Map([[start, {parent: null, move: null}]]);
  const queue = [state.robot];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head], current = pkey(y, x);
    for (const [move, [dy, dx]] of DIRECTION_ENTRIES) {
      const next = pkey(y + dy, x + dx);
      if (parents.has(next) || !board.floor.has(next) || occupied.has(next)) continue;
      parents.set(next, {parent: current, move});
      queue.push([y + dy, x + dx]);
    }
  }
  return {
    has: position => parents.has(position),
    get: position => {
      const path = [];
      let current = position;
      while (parents.has(current) && parents.get(current).parent !== null) {
        const record = parents.get(current);
        path.unshift(record.move);
        current = record.parent;
      }
      return path;
    },
    keys: () => parents.keys(),
    size: parents.size,
  };
}

function denseOccupancy(state, board) {
  return denseBoxLayout(state.boxes, board).indexByCell;
}

function reachablePaths(state, board) {
  const started = now();
  board.metrics.reachabilityCalls++;
  const {dense} = board, occupied = denseOccupancy(state, board);
  const start = dense.idByKey.get(pkey(state.robot[0], state.robot[1]));
  const parents = new Int32Array(dense.keys.length), parentMoves = new Int8Array(dense.keys.length);
  parents.fill(-2);
  parents[start] = -1;
  const queue = new Int32Array(dense.keys.length);
  queue[0] = start;
  let tail = 1, regionId = start;
  for (let head = 0; head < tail; head++) {
    const current = queue[head];
    for (let direction = 0; direction < DIRECTION_ENTRIES.length; direction++) {
      const next = dense.neighbors[current * DIRECTION_ENTRIES.length + direction];
      if (next < 0 || parents[next] !== -2 || occupied[next] >= 0) continue;
      parents[next] = current;
      parentMoves[next] = direction;
      queue[tail++] = next;
      regionId = Math.min(regionId, next);
    }
  }
  const pathToId = id => {
    if (id === undefined || id < 0 || parents[id] === -2) return [];
    const path = [];
    for (let current = id; parents[current] !== -1; current = parents[current]) {
      path.push(DIRECTION_ENTRIES[parentMoves[current]][0]);
    }
    path.reverse();
    return path;
  };
  board.metrics.reachabilityCells += tail;
  board.metrics.reachabilityMs += now() - started;
  return {
    has: position => {
      const id = dense.idByKey.get(position);
      return id !== undefined && parents[id] !== -2;
    },
    hasId: id => id >= 0 && parents[id] !== -2,
    get: position => pathToId(dense.idByKey.get(position)),
    getId: pathToId,
    keys: function* () {
      for (let index = 0; index < tail; index++) yield dense.keys[queue[index]];
    },
    size: tail,
    occupied,
    board,
    regionId,
  };
}

function minimumBlockerRoutes(reachable, board) {
  const {dense} = board, size = dense.keys.length;
  const distances = new Int16Array(size), parents = new Int32Array(size);
  const settled = new Uint8Array(size);
  distances.fill(32767);
  parents.fill(-1);
  for (const position of reachable.keys()) distances[dense.idByKey.get(position)] = 0;
  for (let count = 0; count < size; count++) {
    let current = -1, best = 32767;
    for (let id = 0; id < size; id++) {
      if (!settled[id] && distances[id] < best) {
        current = id;
        best = distances[id];
      }
    }
    if (current < 0) break;
    settled[current] = 1;
    for (let direction = 0; direction < DIRECTION_ENTRIES.length; direction++) {
      const next = dense.neighbors[current * DIRECTION_ENTRIES.length + direction];
      if (next < 0 || settled[next]) continue;
      const distance = best + (reachable.occupied[next] >= 0 ? 1 : 0);
      if (distance >= distances[next]) continue;
      distances[next] = distance;
      parents[next] = current;
    }
  }
  const routeTo = destination => {
    if (destination < 0 || distances[destination] === 32767) return null;
    const route = [], blockers = [];
    for (let current = destination; current >= 0; current = parents[current]) {
      route.push(dense.keys[current]);
      if (reachable.occupied[current] >= 0) blockers.push(dense.keys[current]);
    }
    route.reverse();
    blockers.reverse();
    return {route, blockers, blockerCount: distances[destination]};
  };
  return {routeTo};
}

function supportDependencyGraph(state, board, reachable = reachablePaths(state, board)) {
  const metrics = board.metrics;
  metrics.supportDependencyCalls++;
  const key = `${reachable.regionId}|${boxSignature(state.boxes, board)}`;
  if (board.supportDependencyMemo.has(key)) {
    metrics.supportDependencyCacheHits++;
    return board.supportDependencyMemo.get(key);
  }
  const started = now(), routes = minimumBlockerRoutes(reachable, board);
  const nodes = [], supportDemand = new Map(), prerequisiteDemand = new Map();
  let penalty = 0, optionCount = 0;
  for (const [y, x, label] of state.boxes) {
    const box = pkey(y, x);
    if (board.goals.get(box) === label) continue;
    const targets = board.goalsByLabel.get(label) || [];
    let bestDistance = Infinity;
    for (const target of targets) {
      bestDistance = Math.min(
        bestDistance,
        board.pushDistances.get(target)?.get(box) ?? Infinity,
      );
    }
    if (!Number.isFinite(bestDistance) || bestDistance <= 0) continue;
    const options = [], seen = new Set();
    for (const target of targets) {
      const targetDistances = board.pushDistances.get(target);
      if ((targetDistances?.get(box) ?? Infinity) !== bestDistance) continue;
      for (let direction = 0; direction < DIRECTION_ENTRIES.length; direction++) {
        const [move, [dy, dx]] = DIRECTION_ENTRIES[direction];
        const destination = pkey(y + dy, x + dx);
        if ((targetDistances.get(destination) ?? Infinity) !== bestDistance - 1) continue;
        const support = pkey(y - dy, x - dx);
        const supportId = board.dense.idByKey.get(support);
        if (supportId === undefined || seen.has(`${support}|${destination}`)) continue;
        seen.add(`${support}|${destination}`);
        const accessible = reachable.hasId(supportId);
        const route = accessible
          ? {route: [support], blockers: [], blockerCount: 0}
          : routes.routeTo(supportId);
        options.push({target, destination, support, move, accessible, ...route});
      }
    }
    if (!options.length) continue;
    optionCount += options.length;
    const available = options.some(option => option.accessible);
    const viable = options.filter(option => option.blockerCount !== undefined);
    const minimumBlockers = viable.length
      ? Math.min(...viable.map(option => option.blockerCount))
      : Infinity;
    const preferred = available
      ? options.filter(option => option.accessible)
      : viable.filter(option => option.blockerCount === minimumBlockers);
    const share = 1 / Math.max(1, preferred.length);
    for (const option of preferred) {
      supportDemand.set(option.support, (supportDemand.get(option.support) || 0) + share);
      for (const blocker of option.blockers || []) {
        prerequisiteDemand.set(blocker, (prerequisiteDemand.get(blocker) || 0) + share);
      }
    }
    penalty += available ? 0 : 1 + (Number.isFinite(minimumBlockers) ? minimumBlockers : 4);
    nodes.push({box, label, distance: bestDistance, available, options: preferred});
  }
  const graph = {nodes, supportDemand, prerequisiteDemand, penalty};
  metrics.supportDependencyOptions += optionCount;
  metrics.supportDependencyBlockers += prerequisiteDemand.size;
  metrics.supportDependencyMs += now() - started;
  return memoizeBounded(
    board.supportDependencyMemo,
    key,
    graph,
    SUPPORT_DEPENDENCY_MEMO_LIMIT,
  );
}

function supportDependencyDelta(graph, next) {
  const destroysAccess = graph.supportDemand.get(next.pushedTo) || 0;
  const enablesAccess = graph.prerequisiteDemand.get(next.pushedFrom) || 0;
  return 1.25 * destroysAccess - enablesAccess;
}

function localBoxSignature(boxes) {
  return boxes.map(([position, label]) => `${position},${label}`).sort().join(";");
}

function localReachable(domain, boxes, start) {
  const occupied = new Set(boxes.map(([position]) => position));
  if (!domain.has(start) || occupied.has(start)) return new Set();
  const reached = new Set([start]), queue = [start];
  for (let head = 0; head < queue.length; head++) {
    for (const next of floorNeighbors(queue[head], domain)) {
      if (occupied.has(next) || reached.has(next)) continue;
      reached.add(next);
      queue.push(next);
    }
  }
  return reached;
}

function canonicalLocalState(domain, boxes, robot) {
  const reached = localReachable(domain, boxes, robot);
  if (!reached.size) return null;
  const region = [...reached].sort()[0];
  return {region, signature: `${region}|${localBoxSignature(boxes)}`, reached};
}

function relaxedReversePushTable(board, targetBoxes, maxStates) {
  const states = new Map([[localBoxSignature(targetBoxes), 0]]);
  const queue = [targetBoxes], floor = board.floor;
  let head = 0, cutoff = false;
  for (; head < queue.length && head < maxStates; head++) {
    const current = queue[head], pushes = states.get(localBoxSignature(current));
    const occupied = new Set(current.map(([position]) => position));
    current.forEach(([destination, label], boxIndex) => {
      const [y, x] = destination.split(",").map(Number);
      for (const [, [dy, dx]] of DIRECTION_ENTRIES) {
        const previous = pkey(y - dy, x - dx);
        const support = pkey(y - 2 * dy, x - 2 * dx);
        if (!floor.has(previous) || !floor.has(support) ||
            occupied.has(previous) || occupied.has(support)) continue;
        const predecessor = current.map((box, index) =>
          index === boxIndex ? [previous, label] : box);
        const signature = localBoxSignature(predecessor);
        if (states.has(signature)) continue;
        if (states.size >= maxStates) {
          cutoff = true;
          continue;
        }
        states.set(signature, pushes + 1);
        queue.push(predecessor);
      }
    });
  }
  const complete = !cutoff && head >= queue.length;
  return {status: complete ? "ready" : "cutoff", complete, states, visited: head};
}

function reverseRoomPatternTable(board, room, maxStates = ROOM_PATTERN_MAX_STATES) {
  const roomIndex = board.topology.rooms.indexOf(room);
  if (board.roomPatternTables.has(roomIndex)) return board.roomPatternTables.get(roomIndex);
  const roomLabels = new Set(room.goals.map(goal => board.goals.get(goal)));
  const labels = new Set([...roomLabels].filter(label =>
    (board.goalsByLabel.get(label) || []).every(goal => room.cells.has(goal))));
  const targetBoxes = room.goals
    .filter(goal => labels.has(board.goals.get(goal)))
    .map(goal => [goal, board.goals.get(goal)]);
  if (roomIndex < 0 || targetBoxes.length < 2 || targetBoxes.length > 4) {
    const skipped = {status: "ineligible", complete: false, labels, states: new Map(), visited: 0};
    board.roomPatternTables.set(roomIndex, skipped);
    return skipped;
  }

  // Non-pattern boxes and robot connectivity are removed, while walls, box
  // collisions, labels, and support squares remain. The resulting distance is
  // a lower bound on pushes by these labels in the full puzzle.
  const result = {...relaxedReversePushTable(board, targetBoxes, maxStates), labels};
  board.metrics.roomPatternBuilds++;
  board.metrics.roomPatternStates += result.visited;
  board.roomPatternTables.set(roomIndex, result);
  return result;
}

function roomPatternHeuristicCandidates(boxes, board, assignmentCosts) {
  const candidates = [];
  for (const room of board.topology.rooms) {
    const table = reverseRoomPatternTable(board, room);
    if (!table.states.size) continue;
    const patternBoxes = boxes
      .filter(([, , label]) => table.labels.has(label))
      .map(([y, x, label]) => [pkey(y, x), label]);
    const distance = table.states.get(localBoxSignature(patternBoxes));
    if (distance === undefined) continue;
    board.metrics.roomPatternHits++;
    const assignment = [...table.labels]
      .reduce((total, label) => total + (assignmentCosts.get(label) ?? Infinity), 0);
    const boost = distance - assignment;
    if (boost > 0 && Number.isFinite(boost)) {
      candidates.push({labels: table.labels, boost, kind: "room"});
    }
  }
  return candidates;
}

function shortestPushCriticalCells(position, goal, board) {
  const cacheKey = `${position}>${goal}`;
  if (board.shortestCorridorMemo.has(cacheKey)) return board.shortestCorridorMemo.get(cacheKey);
  const distances = board.pushDistances.get(goal), critical = new Set();
  const initial = distances?.get(position);
  if (!Number.isFinite(initial)) {
    return memoizeBounded(board.shortestCorridorMemo, cacheKey, critical, 10000);
  }
  const seen = new Set([position]), queue = [position];
  for (let head = 0; head < queue.length; head++) {
    const current = queue[head], distance = distances.get(current);
    if (board.topology.articulations.has(current) || board.topology.tunnels.has(current)) {
      critical.add(current);
    }
    if (distance === 0) continue;
    const [y, x] = current.split(",").map(Number);
    for (const [dy, dx] of Object.values(DIRS)) {
      const next = pkey(y + dy, x + dx);
      if (distances.get(next) !== distance - 1 || seen.has(next)) continue;
      seen.add(next);
      queue.push(next);
    }
  }
  return memoizeBounded(board.shortestCorridorMemo, cacheKey, critical, 10000);
}

function reversePairConflictTable(board, leftLabel, rightLabel,
  maxStates = PAIR_CONFLICT_MAX_STATES) {
  const labels = [leftLabel, rightLabel].sort();
  const cacheKey = labels.join("|");
  if (board.pairConflictTables.has(cacheKey)) return board.pairConflictTables.get(cacheKey);
  const goals = labels.map(label => board.goalsByLabel.get(label) || []);
  if (goals.some(entries => entries.length !== 1)) {
    const skipped = {status: "ineligible", complete: false,
      labels: new Set(labels), states: new Map(), visited: 0};
    board.pairConflictTables.set(cacheKey, skipped);
    return skipped;
  }
  const targetBoxes = labels.map((label, index) => [goals[index][0], label]);
  const result = {
    ...relaxedReversePushTable(board, targetBoxes, maxStates),
    labels: new Set(labels),
  };
  board.metrics.pairConflictBuilds++;
  board.metrics.pairConflictStates += result.visited;
  board.pairConflictTables.set(cacheKey, result);
  return result;
}

function pairConflictHeuristicCandidates(boxes, board, assignmentCosts) {
  const byLabel = boxesByLabelWithIndices(boxes);
  const entries = [...byLabel]
    .filter(([label, group]) => group.length === 1 &&
      (board.goalsByLabel.get(label) || []).length === 1)
    .map(([label, [{y, x}]]) => ({label, position: pkey(y, x),
      goal: board.goalsByLabel.get(label)[0]}));
  const candidates = [];
  for (let left = 0; left < entries.length; left++) {
    const leftEntry = entries[left];
    const leftCritical = shortestPushCriticalCells(
      leftEntry.position, leftEntry.goal, board);
    if (!leftCritical.size) continue;
    for (let right = left + 1; right < entries.length; right++) {
      const rightEntry = entries[right];
      const assignment = (assignmentCosts.get(leftEntry.label) ?? Infinity) +
        (assignmentCosts.get(rightEntry.label) ?? Infinity);
      if (assignment > PAIR_CONFLICT_DISTANCE_LIMIT) continue;
      const coveredByRoom = [...board.roomPatternTables.values()].some(table =>
        table.states.size && table.labels.has(leftEntry.label) &&
        table.labels.has(rightEntry.label));
      if (coveredByRoom) continue;
      const rightCritical = shortestPushCriticalCells(
        rightEntry.position, rightEntry.goal, board);
      if (![...leftCritical].some(position => rightCritical.has(position))) continue;
      board.metrics.pairConflictCandidates++;
      const table = reversePairConflictTable(
        board, leftEntry.label, rightEntry.label);
      const patternBoxes = [leftEntry, rightEntry]
        .map(entry => [entry.position, entry.label]);
      const distance = table.states.get(localBoxSignature(patternBoxes));
      if (distance === undefined) continue;
      board.metrics.pairConflictHits++;
      const boost = distance - assignment;
      if (boost > 0 && Number.isFinite(boost)) {
        candidates.push({labels: table.labels, boost, kind: "pair"});
      }
    }
  }
  return candidates;
}

function maximumDisjointPatternSelection(candidates) {
  if (!candidates.length) return [];
  const labels = [...new Set(candidates.flatMap(candidate => [...candidate.labels]))];
  if (labels.length > 20) {
    const selected = [], used = new Set();
    for (const candidate of [...candidates].sort((a, b) => b.boost - a.boost)) {
      if ([...candidate.labels].some(label => used.has(label))) continue;
      selected.push(candidate);
      candidate.labels.forEach(label => used.add(label));
    }
    return selected;
  }
  const labelIndex = new Map(labels.map((label, index) => [label, index]));
  const weighted = candidates.map(candidate => ({
    candidate,
    mask: [...candidate.labels]
      .reduce((mask, label) => mask | (1 << labelIndex.get(label)), 0),
  }));
  const best = new Map([[0, {boost: 0, selected: []}]]);
  for (const {candidate, mask} of weighted) {
    for (const [used, entry] of [...best]) {
      if (used & mask) continue;
      const combined = used | mask, boost = entry.boost + candidate.boost;
      if ((best.get(combined)?.boost ?? -Infinity) >= boost) continue;
      best.set(combined, {boost, selected: [...entry.selected, candidate]});
    }
  }
  return [...best.values()].reduce((left, right) =>
    right.boost > left.boost ? right : left).selected;
}

function interactionHeuristicBoost(boxes, board, assignmentCosts) {
  const selected = maximumDisjointPatternSelection([
    ...roomPatternHeuristicCandidates(boxes, board, assignmentCosts),
    ...pairConflictHeuristicCandidates(boxes, board, assignmentCosts),
  ]);
  const roomBoost = selected.filter(candidate => candidate.kind === "room")
    .reduce((total, candidate) => total + candidate.boost, 0);
  const pairBoost = selected.filter(candidate => candidate.kind === "pair")
    .reduce((total, candidate) => total + candidate.boost, 0);
  board.metrics.roomPatternBoost += roomBoost;
  board.metrics.pairConflictBoost += pairBoost;
  return roomBoost + pairBoost;
}

function reverseGoalRoomPackingTable(board, room, maxStates = 20000) {
  const roomIndex = board.topology.rooms.indexOf(room);
  if (board.goalRoomPackingTables.has(roomIndex)) {
    return board.goalRoomPackingTables.get(roomIndex);
  }
  const domain = new Set([...room.cells, room.gate]);
  for (const [position, distance] of room.approach) if (distance <= 2) domain.add(position);
  const targetBoxes = room.goals.map(position => [position, board.goals.get(position)]);
  if (roomIndex < 0 || room.cells.size > 16 || domain.size > 24 || targetBoxes.length > 4) {
    const skipped = {status: "oversized", complete: false, domain, states: new Map(), visited: 0};
    board.goalRoomPackingTables.set(roomIndex, skipped);
    return skipped;
  }

  const started = now(), states = new Map(), queue = [];
  const targetOccupied = new Set(targetBoxes.map(([position]) => position));
  for (const robot of domain) {
    if (targetOccupied.has(robot)) continue;
    const canonical = canonicalLocalState(domain, targetBoxes, robot);
    if (!canonical || states.has(canonical.signature)) continue;
    const entry = {pushes: 0, nextPushes: new Set()};
    states.set(canonical.signature, entry);
    queue.push({robot: canonical.region, boxes: targetBoxes, pushes: 0});
  }

  let visited = 0, head = 0;
  for (; head < queue.length && visited < maxStates; head++) {
    const current = queue[head];
    const canonical = canonicalLocalState(domain, current.boxes, current.robot);
    if (!canonical) continue;
    visited++;
    const occupied = new Set(current.boxes.map(([position]) => position));
    current.boxes.forEach(([destination, label], boxIndex) => {
      const [y, x] = destination.split(",").map(Number);
      for (const [, [dy, dx]] of DIRECTION_ENTRIES) {
        const previous = pkey(y - dy, x - dx);
        const support = pkey(y - 2 * dy, x - 2 * dx);
        if (!domain.has(previous) || !domain.has(support) ||
            occupied.has(previous) || occupied.has(support) ||
            !canonical.reached.has(previous)) continue;
        const predecessorBoxes = current.boxes.map((box, index) =>
          index === boxIndex ? [previous, label] : box);
        const predecessor = canonicalLocalState(domain, predecessorBoxes, support);
        if (!predecessor) continue;
        const pushes = current.pushes + 1;
        const forwardPush = `${previous}>${destination}`;
        const known = states.get(predecessor.signature);
        if (!known) {
          states.set(predecessor.signature, {pushes, nextPushes: new Set([forwardPush])});
          queue.push({robot: predecessor.region, boxes: predecessorBoxes, pushes});
        } else if (known.pushes === pushes) {
          known.nextPushes.add(forwardPush);
        }
      }
    });
  }
  const complete = head >= queue.length;
  const result = {status: complete ? "ready" : "cutoff", complete, domain, states, visited};
  board.metrics.reversePackingBuilds++;
  board.metrics.reversePackingStates += visited;
  board.metrics.localRoomMs += now() - started;
  board.goalRoomPackingTables.set(roomIndex, result);
  return result;
}

function exactLocalPushSearch({domain, boxes, robot, isGoal, gate, maxStates = 10000}) {
  const initialBoxes = boxes.map(box => [...box]);
  const queue = [{robot, boxes: initialBoxes, pushes: 0, firstPush: null}];
  const seen = new Set(), firstPushes = new Set();
  let visited = 0, viableBoundaries = 0, solutionPushes = Infinity;
  for (let head = 0; head < queue.length; head++) {
    if (visited >= maxStates) break;
    const current = queue[head];
    if (current.pushes > solutionPushes) break;
    const reached = localReachable(domain, current.boxes, current.robot);
    if (!reached.size) continue;
    const region = [...reached].sort()[0];
    const signature = `${region}|${localBoxSignature(current.boxes)}`;
    if (seen.has(signature)) continue;
    seen.add(signature);
    visited++;
    const occupied = new Map(current.boxes.map(([position, label], index) =>
      [position, {label, index}]));
    if (!occupied.has(gate) && reached.has(gate)) viableBoundaries++;
    if (isGoal(occupied, reached)) {
      solutionPushes = current.pushes;
      if (current.firstPush) firstPushes.add(current.firstPush);
      continue;
    }
    if (current.pushes >= solutionPushes) continue;
    current.boxes.forEach(([position, label], index) => {
      const [y, x] = position.split(",").map(Number);
      for (const [, [dy, dx]] of DIRECTION_ENTRIES) {
        const support = pkey(y - dy, x - dx);
        const destination = pkey(y + dy, x + dx);
        if (!reached.has(support) || !domain.has(destination) || occupied.has(destination)) continue;
        const nextBoxes = current.boxes.map((box, boxIndex) =>
          boxIndex === index ? [destination, label] : box);
        const push = `${position}>${destination}`;
        queue.push({
          robot: position,
          boxes: nextBoxes,
          pushes: current.pushes + 1,
          firstPush: current.firstPush || push,
        });
      }
    });
  }
  const cutoff = visited >= maxStates && !Number.isFinite(solutionPushes);
  return {
    status: Number.isFinite(solutionPushes) ? (solutionPushes === 0 ? "packed" : "solvable")
      : cutoff ? "cutoff" : "exhausted",
    pushes: Number.isFinite(solutionPushes) ? solutionPushes : null,
    firstPushes,
    visited,
    viableBoundaries,
  };
}

function exactLocalRoomSearch(state, board, room, reachable = reachablePaths(state, board)) {
  const metrics = board.metrics;
  metrics.localRoomCalls++;
  const roomIndex = board.topology.rooms.indexOf(room);
  const domain = new Set([...room.cells, room.gate]);
  for (const [position, distance] of room.approach) if (distance <= 2) domain.add(position);
  const localBoxes = state.boxes
    .map(([y, x, label]) => [pkey(y, x), label])
    .filter(([position]) => domain.has(position));
  const entries = [...domain].filter(position => reachable.has(position)).sort();
  const entrySide = entries[0] || "sealed";
  const cacheKey = `${roomIndex}|${entrySide}|${localBoxSignature(localBoxes)}`;
  if (board.localRoomMemo.has(cacheKey)) {
    metrics.localRoomCacheHits++;
    return board.localRoomMemo.get(cacheKey);
  }
  const started = now();
  const internalCounts = new Map(), localCounts = new Map(), targetCounts = new Map();
  localBoxes.forEach(([position, label]) => {
    localCounts.set(label, (localCounts.get(label) || 0) + 1);
    if (room.cells.has(position)) internalCounts.set(label, (internalCounts.get(label) || 0) + 1);
  });
  room.goals.forEach(goal => {
    const label = board.goals.get(goal);
    targetCounts.set(label, (targetCounts.get(label) || 0) + 1);
  });
  let importsRequired = 0, exportsRequired = 0, missingLocalBox = false;
  for (const [label, count] of targetCounts) {
    importsRequired += Math.max(0, count - (internalCounts.get(label) || 0));
    if ((localCounts.get(label) || 0) < count) missingLocalBox = true;
  }
  if (missingLocalBox) {
      const result = {
        status: "needs-import", importsRequired, exportsRequired,
        doorwayOccupied: localBoxes.some(([position]) => position === room.gate),
        entrySide, domain, roomIndex, firstPushes: new Set(), visited: 0, viableBoundaries: 0,
      };
      metrics.localRoomMs += now() - started;
    return memoizeBounded(board.localRoomMemo, cacheKey, result, LOCAL_ROOM_MEMO_LIMIT);
  }
  for (const [label, count] of internalCounts) {
    exportsRequired += Math.max(0, count - (targetCounts.get(label) || 0));
  }
  if (room.cells.size > 16 || domain.size > 24 || localBoxes.length > 5) {
    const result = {
      status: "oversized", importsRequired, exportsRequired,
      doorwayOccupied: localBoxes.some(([position]) => position === room.gate),
      entrySide, domain, roomIndex, firstPushes: new Set(), visited: 0, viableBoundaries: 0,
    };
    metrics.localRoomMs += now() - started;
    return memoizeBounded(board.localRoomMemo, cacheKey, result, LOCAL_ROOM_MEMO_LIMIT);
  }
  if (!entries.length) {
    const result = {
      status: "inaccessible", importsRequired, exportsRequired,
      doorwayOccupied: localBoxes.some(([position]) => position === room.gate),
      entrySide, domain, roomIndex, firstPushes: new Set(), visited: 0, viableBoundaries: 0,
    };
    metrics.localRoomMs += now() - started;
      return memoizeBounded(board.localRoomMemo, cacheKey, result, LOCAL_ROOM_MEMO_LIMIT);
  }
  const packingTable = reverseGoalRoomPackingTable(board, room);
  const localCanonical = canonicalLocalState(domain, localBoxes, entries[0]);
  const hasSingleLocalEntryRegion = localCanonical &&
    entries.every(position => localCanonical.reached.has(position));
  const tableEntry = localBoxes.length === room.goals.length && exportsRequired === 0 &&
    hasSingleLocalEntryRegion ? packingTable.states.get(localCanonical.signature) : null;
  if (tableEntry) {
    metrics.reversePackingHits++;
    const result = {
      status: tableEntry.pushes === 0 ? "packed" : "solvable",
      pushes: tableEntry.pushes,
      firstPushes: new Set(tableEntry.nextPushes),
      visited: 0,
      viableBoundaries: entries.includes(room.gate) ? 1 : 0,
      importsRequired,
      exportsRequired,
      doorwayOccupied: localBoxes.some(([position]) => position === room.gate),
      entrySide,
      domain,
      roomIndex,
      source: "reverse-packing-table",
      reverseTableComplete: packingTable.complete,
    };
    metrics.localRoomMs += now() - started;
    return memoizeBounded(board.localRoomMemo, cacheKey, result, LOCAL_ROOM_MEMO_LIMIT);
  }
  const search = exactLocalPushSearch({
    domain,
    boxes: localBoxes,
    robot: entries[0],
    gate: room.gate,
    isGoal: occupied => !occupied.has(room.gate) && room.goals.every(goal =>
      occupied.get(goal)?.label === board.goals.get(goal)) &&
      [...occupied].every(([position, {label}]) =>
        !room.cells.has(position) || board.goals.get(position) === label),
  });
  const result = {
    ...search,
    importsRequired,
    exportsRequired,
    doorwayOccupied: localBoxes.some(([position]) => position === room.gate),
    entrySide,
    domain,
    roomIndex,
  };
  metrics.localRoomStates += search.visited;
  metrics.localRoomMs += now() - started;
  return memoizeBounded(board.localRoomMemo, cacheKey, result, LOCAL_ROOM_MEMO_LIMIT);
}

function exactLocalRoomAnalyses(state, board, reachable = reachablePaths(state, board)) {
  return board.topology.rooms.map(room => exactLocalRoomSearch(state, board, room, reachable));
}

function localRoomOrderingDelta(analyses, next) {
  const push = `${next.pushedFrom}>${next.pushedTo}`;
  let delta = 0;
  for (const analysis of analyses) {
    if (analysis.status !== "solvable") continue;
    const confidence = analysis.kind === "corral" ? 0.2 : 1;
    if (analysis.firstPushes.has(push)) delta -= confidence;
    else if (analysis.domain.has(next.pushedFrom) || analysis.domain.has(next.pushedTo)) {
      delta += 0.2 * confidence;
    }
  }
  return delta;
}

function inaccessibleFloorComponents(reachable, board) {
  const inaccessible = new Set([...board.floor].filter(position => !reachable.has(position)));
  const components = [];
  while (inaccessible.size) {
    const start = inaccessible.values().next().value;
    const component = new Set([start]), queue = [start];
    inaccessible.delete(start);
    for (let head = 0; head < queue.length; head++) {
      for (const next of floorNeighbors(queue[head], board.floor)) {
        if (!inaccessible.has(next)) continue;
        inaccessible.delete(next);
        component.add(next);
        queue.push(next);
      }
    }
    components.push(component);
  }
  return components;
}

function exactLocalCorralSearch(state, board, component, reachable) {
  const metrics = board.metrics;
  metrics.localCorralCalls++;
  const occupied = new Map(state.boxes.map(([y, x, label]) => [pkey(y, x), label]));
  const componentBoxes = [...component].filter(position => occupied.has(position));
  if (componentBoxes.length === component.size) return null;
  if (!componentBoxes.some(position => board.goals.get(position) !== occupied.get(position))) {
    return null;
  }
  const boundary = new Set();
  for (const position of component) {
    for (const next of floorNeighbors(position, board.floor)) {
      if (reachable.has(next)) boundary.add(next);
    }
  }
  const domain = new Set([...component, ...boundary]);
  const localBoxes = state.boxes
    .map(([y, x, label]) => [pkey(y, x), label])
    .filter(([position]) => domain.has(position));
  const entrySide = [...boundary].sort()[0] || "sealed";
  const componentKey = [...component].sort().join(".");
  const cacheKey = `${componentKey}|${entrySide}|${localBoxSignature(localBoxes)}`;
  if (board.localCorralMemo.has(cacheKey)) {
    metrics.localCorralCacheHits++;
    return board.localCorralMemo.get(cacheKey);
  }
  const started = now();
  if (!boundary.size || component.size > 16 || domain.size > 24 || localBoxes.length > 5) {
    const result = {
      kind: "corral",
      status: !boundary.size ? "sealed" : "oversized",
      domain,
      firstPushes: new Set(),
      visited: 0,
      viableBoundaries: 0,
    };
    metrics.localCorralMs += now() - started;
    return memoizeBounded(board.localCorralMemo, cacheKey, result, LOCAL_CORRAL_MEMO_LIMIT);
  }
  const componentGoals = [...component].filter(position => board.goals.has(position));
  const search = exactLocalPushSearch({
    domain,
    boxes: localBoxes,
    robot: entrySide,
    gate: entrySide,
    isGoal: (localOccupied, reached) =>
      [...reached].some(position => component.has(position)) ||
      (componentGoals.every(goal =>
        localOccupied.get(goal)?.label === board.goals.get(goal)) &&
        [...localOccupied].every(([position, {label}]) =>
          !component.has(position) || board.goals.get(position) === label)),
  });
  const result = {...search, kind: "corral", domain};
  metrics.localCorralStates += search.visited;
  metrics.localCorralMs += now() - started;
  return memoizeBounded(board.localCorralMemo, cacheKey, result, LOCAL_CORRAL_MEMO_LIMIT);
}

function exactLocalCorralAnalyses(state, board, reachable = reachablePaths(state, board)) {
  return inaccessibleFloorComponents(reachable, board)
    .map(component => exactLocalCorralSearch(state, board, component, reachable))
    .filter(Boolean);
}

function createsSealedCorralDeadlock(state, board, reachable) {
  const occupied = new Map(state.boxes.map(([y, x, label]) => [pkey(y, x), label]));
  for (const component of inaccessibleFloorComponents(reachable, board)) {
    const componentBoxes = [...component].filter(position => occupied.has(position));
    if (!componentBoxes.some(position => board.goals.get(position) !== occupied.get(position))) continue;
    const canOpen = componentBoxes.some(position => {
      const [y, x] = position.split(",").map(Number);
      return Object.values(DIRS).some(([dy, dx]) => {
        const support = pkey(y - dy, x - dx), destination = pkey(y + dy, x + dx);
        return reachable.has(support) && board.floor.has(destination) && !occupied.has(destination);
      });
    });
    if (!canOpen) return true;
  }
  return false;
}

// This registry is deliberately executable production data. Differential tests
// enumerate it, so a new hard prune cannot be added without naming an independent
// oracle family that certifies it against unpruned reachability.
const HARD_PRUNING_RULES = Object.freeze([
  Object.freeze({name: "static-dead", oracleFamily: "push-reachability",
    scope: "push", detect: (boxes, board, movedBox, label) =>
      staticDead(movedBox[0], movedBox[1], board, label)}),
  Object.freeze({name: "2x2", oracleFamily: "multi-box-local",
    scope: "dynamic", detect: creates2x2Deadlock}),
  Object.freeze({name: "closed-diagonal", oracleFamily: "closed-diagonal",
    scope: "dynamic", detect: createsClosedDiagonalDeadlock}),
  Object.freeze({name: "freeze", oracleFamily: "interacting-freeze",
    scope: "dynamic", detect: createsFrozenComponentDeadlock}),
  Object.freeze({name: "pattern-database", oracleFamily: "typed-corridor",
    scope: "dynamic", detect: createsPatternDatabaseDeadlock}),
  Object.freeze({name: "sealed-corral", oracleFamily: "corral",
    scope: "state", detect: createsSealedCorralDeadlock}),
  Object.freeze({name: "proven-commitment", oracleFamily: "goal-commitment",
    scope: "push-neighbor", detect: null}),
]);
const DYNAMIC_HARD_PRUNING_RULES = HARD_PRUNING_RULES.filter(
  rule => rule.scope === "dynamic",
);
globalThis.SokomindHardPruningRules = HARD_PRUNING_RULES;

function pushNeighbors(state, board, reachable = reachablePaths(state, board), options = {}) {
  board.metrics.pushNeighborCalls++;
  const occupied = reachable.occupied || denseOccupancy(state, board);
  const commitments = options.commitments || (options.lockProven && board.topology.rooms.length
    ? stateGoalCommitments(state, board, reachable)
    : null);
  const result = [];
  state.boxes.forEach(([y, x, label], index) => {
    const boxPosition = pkey(y, x);
    if (commitments?.get(boxPosition) === GOAL_COMMITMENT.PROVEN) {
      board.metrics.commitmentBoxLocks++;
      return;
    }
    const boxId = board.dense.idByKey.get(boxPosition);
    for (let direction = 0; direction < DIRECTION_ENTRIES.length; direction++) {
      const [move] = DIRECTION_ENTRIES[direction];
      const supportId = board.dense.neighbors[
        boxId * DIRECTION_ENTRIES.length + OPPOSITE_DIRECTION_INDEX[direction]
      ];
      const destinationId = board.dense.neighbors[boxId * DIRECTION_ENTRIES.length + direction];
      if (!reachable.hasId(supportId) || destinationId < 0 || occupied[destinationId] >= 0) continue;
      const destinationY = board.dense.y[destinationId], destinationX = board.dense.x[destinationId];
      const dest = board.dense.keys[destinationId];
      board.metrics.pushCandidates++;
      const boxes = state.boxes.slice();
      boxes[index] = [destinationY, destinationX, label];
      if (staticDead(destinationY, destinationX, board, label)) {
        board.metrics.staticDeadPrunes++;
        continue;
      }
      deriveDenseBoxLayout(state.boxes, boxes, index, destinationId, board);
      if (createsDynamicDeadlock(boxes, board, [destinationY, destinationX])) {
        board.metrics.dynamicDeadPrunes++;
        continue;
      }
      board.assignmentParentMemo.set(boxes, {parentBoxes: state.boxes, changedIndex: index});
      result.push({
        robot: [y, x],
        boxes,
        path: [...reachable.getId(supportId), move],
        pushClass: `${label}:${y},${x}:${move}`,
        pushedFrom: pkey(y, x),
        pushedTo: dest,
      });
      board.metrics.pushesRetained++;
    }
  });
  return result;
}

function pushBoxNeighbors(
  state,
  board,
  boxPosition,
  reachable = reachablePaths(state, board),
  options = {},
) {
  const occupied = reachable.occupied || denseOccupancy(state, board);
  const boxId = board.dense.idByKey.get(boxPosition);
  if (boxId === undefined || occupied[boxId] < 0) return [];
  const commitments = options.commitments || (options.lockProven && board.topology.rooms.length
    ? stateGoalCommitments(state, board, reachable)
    : null);
  if (commitments?.get(boxPosition) === GOAL_COMMITMENT.PROVEN) {
    board.metrics.commitmentBoxLocks++;
    return [];
  }
  const index = occupied[boxId];
  const [y, x, label] = state.boxes[index], result = [];
  for (let direction = 0; direction < DIRECTION_ENTRIES.length; direction++) {
    const [move] = DIRECTION_ENTRIES[direction];
    const supportId = board.dense.neighbors[
      boxId * DIRECTION_ENTRIES.length + OPPOSITE_DIRECTION_INDEX[direction]
    ];
    const destinationId = board.dense.neighbors[boxId * DIRECTION_ENTRIES.length + direction];
    if (!reachable.hasId(supportId) || destinationId < 0 || occupied[destinationId] >= 0) continue;
    const destinationY = board.dense.y[destinationId], destinationX = board.dense.x[destinationId];
    const destination = board.dense.keys[destinationId];
    const boxes = state.boxes.slice();
    boxes[index] = [destinationY, destinationX, label];
    if (staticDead(destinationY, destinationX, board, label)) continue;
    deriveDenseBoxLayout(state.boxes, boxes, index, destinationId, board);
    if (createsDynamicDeadlock(boxes, board, [destinationY, destinationX])) continue;
    board.assignmentParentMemo.set(boxes, {parentBoxes: state.boxes, changedIndex: index});
    result.push({
      robot: [y, x],
      boxes,
      path: [...reachable.getId(supportId), move],
      pushClass: `${label}:${y},${x}:${move}`,
      pushedFrom: boxPosition,
      pushedTo: destination,
    });
  }
  return result;
}

function pushKey(state, reachable) {
  if (reachable.board && reachable.regionId !== undefined) {
    return `${reachable.regionId.toString(36)}|${boxSignature(state.boxes, reachable.board)}`;
  }
  let robotRegion = null;
  for (const position of reachable.keys()) {
    if (robotRegion === null || position < robotRegion) robotRegion = position;
  }
  return `${robotRegion}|${[...state.boxes.map(b => b.join(","))].sort().join(";")}`;
}

function collapseForcedPushes(first, board, limit = 32, options = {}) {
  let state = {robot: first.robot, boxes: first.boxes};
  const path = [...first.path], seen = new Set([exactPushKey(state, board)]);
  let pushes = 1;
  while (pushes < limit && !goal(state.boxes, board.goals)) {
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) return null;
    const choices = pushNeighbors(state, board, reachable, {lockProven: options.lockProven});
    if (choices.length !== 1) break;
    const next = choices[0], signature = exactPushKey(next, board);
    if (seen.has(signature)) break;
    seen.add(signature);
    path.push(...next.path);
    state = {robot: next.robot, boxes: next.boxes};
    pushes++;
  }
  return {...state, path, pushes, pushClass: first.pushClass};
}

function expandPushMacro(next, board, enabled = true, options = {}) {
  if (!enabled || !board.topology.tunnels.has(next.pushedTo)) return {...next, pushes: 1};
  return collapseForcedPushes(next, board, 32, options);
}

function expandPushSequences(
  first,
  board,
  maxPushes = 12,
  maxExplored = 48,
  maxReturned = 8,
  options = {},
) {
  const initial = {...first, pushes: 1};
  const queue = [initial], endpoints = [];
  const seen = new Set([exactPushKey(initial, board)]);
  let head = 0;
  for (; head < queue.length && queue.length < maxExplored; head++) {
    const current = queue[head];
    if (current.pushes >= maxPushes || goal(current.boxes, board.goals)) {
      endpoints.push(current);
      continue;
    }
    const state = {robot: current.robot, boxes: current.boxes};
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) {
      endpoints.push(current);
      continue;
    }
    const continuations = pushBoxNeighbors(
      state,
      board,
      current.pushedTo,
      reachable,
      {lockProven: options.lockProven},
    );
    if (!continuations.length) endpoints.push({...current, macroDecision: true});
    else if (continuations.length > 1) endpoints.push({...current, macroDecision: true});
    for (const next of continuations) {
      const sequence = {
        robot: next.robot,
        boxes: next.boxes,
        path: [...current.path, ...next.path],
        pushes: current.pushes + 1,
        pushClass: `${first.pushClass}:${current.pushes + 1}`,
        pushedFrom: next.pushedFrom,
        pushedTo: next.pushedTo,
      };
      const signature = exactPushKey(sequence, board);
      if (seen.has(signature)) continue;
      seen.add(signature);
      queue.push(sequence);
      if (queue.length >= maxExplored) break;
    }
  }
  endpoints.push(...queue.slice(head));
  endpoints.sort((left, right) =>
    Number(Boolean(right.macroDecision)) - Number(Boolean(left.macroDecision)) ||
    right.pushes - left.pushes);
  const selected = [], destinations = new Set();
  for (const endpoint of endpoints) {
    if (destinations.has(endpoint.pushedTo)) continue;
    destinations.add(endpoint.pushedTo);
    selected.push(endpoint);
    if (selected.length >= maxReturned) break;
  }
  return [initial, ...selected.filter(endpoint =>
    exactPushKey(endpoint, board) !== exactPushKey(initial, board))];
}

function expandStraightPushes(first, board, maxPushes = 8, options = {}) {
  const [fromY, fromX] = first.pushedFrom.split(",").map(Number);
  const [toY, toX] = first.pushedTo.split(",").map(Number);
  const dy = toY - fromY, dx = toX - fromX;
  const results = [{...first, pushes: 1}];
  let current = results[0];
  while (current.pushes < maxPushes && !goal(current.boxes, board.goals)) {
    const state = {robot: current.robot, boxes: current.boxes};
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) break;
    const [y, x] = current.pushedTo.split(",").map(Number);
    const destination = pkey(y + dy, x + dx);
    const next = pushBoxNeighbors(
      state,
      board,
      current.pushedTo,
      reachable,
      {lockProven: options.lockProven},
    )
      .find(candidate => candidate.pushedTo === destination);
    if (!next) break;
    current = {
      ...next,
      path: [...current.path, ...next.path],
      pushes: current.pushes + 1,
      pushClass: `${first.pushClass}:${current.pushes + 1}`,
    };
    results.push(current);
  }
  return results;
}

function invertWalk(path) {
  return [...path].reverse().map(move => OPPOSITE[move]);
}
function encodeMoves(path) {
  return path.map(move => MOVE_CODE[move]).join("");
}

function reversePullNeighbors(state, board, reachable = reachablePaths(state, board)) {
  const occupied = reachable.occupied || denseOccupancy(state, board);
  const result = [];
  state.boxes.forEach(([y, x, label], index) => {
    const boxId = board.dense.idByKey.get(pkey(y, x));
    for (let direction = 0; direction < DIRECTION_ENTRIES.length; direction++) {
      const [move] = DIRECTION_ENTRIES[direction];
      const opposite = OPPOSITE_DIRECTION_INDEX[direction];
      const boxBeforeId = board.dense.neighbors[boxId * DIRECTION_ENTRIES.length + opposite];
      if (!reachable.hasId(boxBeforeId) || occupied[boxBeforeId] >= 0) continue;
      const robotAfterPullId = board.dense.neighbors[
        boxBeforeId * DIRECTION_ENTRIES.length + opposite
      ];
      if (robotAfterPullId < 0 || occupied[robotAfterPullId] >= 0) continue;
      const boxY = board.dense.y[boxBeforeId], boxX = board.dense.x[boxBeforeId];
      if (staticDead(boxY, boxX, board, label)) continue;
      const boxes = state.boxes.slice();
      boxes[index] = [boxY, boxX, label];
      deriveDenseBoxLayout(state.boxes, boxes, index, boxBeforeId, board);
      if (creates2x2Deadlock(boxes, board, [boxY, boxX]) ||
          createsFrozenComponentDeadlock(boxes, board, [boxY, boxX])) continue;
      const walkToPullSpot = reachable.getId(boxBeforeId);
      const walkFromPushLanding = invertWalk(walkToPullSpot);
      result.push({
        robot: [board.dense.y[robotAfterPullId], board.dense.x[robotAfterPullId]],
        boxes,
        cost: state.cost + 1,
        segment: [move, ...walkFromPushLanding],
      });
    }
  });
  return result;
}

function solvedBoxes(board, initialBoxes) {
  const byLabel = new Map();
  initialBoxes.forEach(([, , label]) => byLabel.set(label, (byLabel.get(label) || 0) + 1));
  const boxes = [];
  for (const [label, count] of byLabel) {
    const goals = [...board.goals]
      .filter(([, goalLabel]) => goalLabel === label)
      .map(([position]) => position.split(",").map(Number))
      .slice(0, count);
    goals.forEach(([y, x]) => boxes.push([y, x, label]));
  }
  return boxes.sort((a, b) => a.join(",").localeCompare(b.join(",")));
}

function reverseStartPortfolio(board, initialBoxes, initialTargets = null) {
  const boxes = solvedBoxes(board, initialBoxes);
  const occupied = new Set(boxes.map(([y, x]) => pkey(y, x)));
  const unique = new Map();
  for (const position of board.floor) {
    if (occupied.has(position)) continue;
    const robot = position.split(",").map(Number);
    const state = {robot, boxes, cost: 0};
    const reachable = reachablePaths(state, board);
    const signature = pushKey(state, reachable);
    if (!unique.has(signature)) unique.set(signature, {state, signature, reachable});
  }
  const targets = initialTargets || targetMapFromBoxes(initialBoxes, board);
  return [...unique.values()].map(entry => {
    const pulls = reversePullNeighbors(entry.state, board, entry.reachable);
    const nextEstimate = pulls.reduce((best, next) =>
      Math.min(best, homeHeuristic(next.boxes, targets)), Infinity);
    const gateAccess = [...board.topology.articulations]
      .filter(position => entry.reachable.has(position)).length;
    return {
      ...entry,
      pullOptions: pulls.length,
      pullSignatures: pulls.map(next => exactPushKey(next, board)),
      nextEstimate,
      reachableCells: entry.reachable.size,
      gateAccess,
    };
  }).sort((left, right) =>
    Number(right.pullOptions > 0) - Number(left.pullOptions > 0) ||
    right.pullOptions - left.pullOptions ||
    left.nextEstimate - right.nextEstimate ||
    right.gateAccess - left.gateAccess ||
    right.reachableCells - left.reachableCells ||
    left.signature.localeCompare(right.signature));
}

function reverseStartStates(board, initialBoxes, shard, initialTargets = null) {
  const portfolio = reverseStartPortfolio(board, initialBoxes, initialTargets);
  const states = portfolio.map(entry => entry.state);
  const assignedPullOptions = portfolio.reduce((sum, entry) => sum +
    entry.pullSignatures.filter(signature => reverseShardOwns(signature, shard)).length, 0);
  states.portfolioStats = {
    totalRegions: portfolio.length,
    productiveRegions: portfolio.filter(entry => entry.pullOptions > 0).length,
    totalPullOptions: portfolio.reduce((sum, entry) => sum + entry.pullOptions, 0),
    assignedRegions: states.length,
    assignedProductiveRegions: portfolio.filter(entry => entry.pullSignatures
      .some(signature => reverseShardOwns(signature, shard))).length,
    assignedPullOptions,
  };
  return states;
}

function reverseShardOwns(signature, shard) {
  if (!shard || shard.count <= 1) return true;
  return Math.floor(signatureNoise(signature, 0x51f15e) * shard.count) === shard.index;
}
