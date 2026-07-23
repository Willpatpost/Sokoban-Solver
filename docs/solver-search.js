function flushRecords(records) {
  if (records.length) postMessage({type: "records", records: records.splice(0, records.length)});
}

function reconstructPath(cameFrom, signature) {
  const path = [];
  let current = signature;
  while (cameFrom.has(current)) {
    const {parent, segment} = cameFrom.get(current);
    path.unshift(...segment);
    current = parent;
  }
  return path;
}

function signatureNoise(signature, seed) {
  let hash = (2166136261 ^ seed) >>> 0;
  if (typeof signature === "bigint") {
    let value = signature;
    do {
      hash ^= Number(value & 0xffffffffn);
      hash = Math.imul(hash, 16777619) >>> 0;
      value >>= 32n;
    } while (value);
    return hash / 0x100000000;
  }
  for (let index = 0; index < signature.length; index++) {
    hash ^= signature.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash / 0x100000000;
}

function reconstructNodePath(node) {
  const segments = [];
  for (let current = node; current; current = current.parent) segments.push(current.segment);
  const path = [];
  for (let index = segments.length - 1; index >= 0; index--) path.push(...segments[index]);
  return path;
}

function serializeSearchCheckpoint(candidate, board) {
  if (!candidate) return null;
  return {
    state: {
      rows: board.rows,
      robot: candidate.robot,
      boxes: candidate.boxes.map(([y, x, label]) => [pkey(y, x), label]),
    },
    path: reconstructNodePath(candidate.node),
    cost: candidate.cost,
    estimate: candidate.estimate,
  };
}

const PLAN_BOARD_TRANSFORMS = [
  {
    id: "identity",
    dimensions: (height, width) => [height, width],
    forward: (_height, _width, y, x) => [y, x],
    inverse: (_height, _width, y, x) => [y, x],
  },
  {
    id: "mirror-horizontal",
    dimensions: (height, width) => [height, width],
    forward: (_height, width, y, x) => [y, width - 1 - x],
    inverse: (_height, width, y, x) => [y, width - 1 - x],
  },
  {
    id: "mirror-vertical",
    dimensions: (height, width) => [height, width],
    forward: (height, _width, y, x) => [height - 1 - y, x],
    inverse: (height, _width, y, x) => [height - 1 - y, x],
  },
  {
    id: "rotate-180",
    dimensions: (height, width) => [height, width],
    forward: (height, width, y, x) => [height - 1 - y, width - 1 - x],
    inverse: (height, width, y, x) => [height - 1 - y, width - 1 - x],
  },
  {
    id: "rotate-90",
    dimensions: (height, width) => [width, height],
    forward: (height, _width, y, x) => [x, height - 1 - y],
    inverse: (height, _width, y, x) => [height - 1 - x, y],
  },
  {
    id: "rotate-270",
    dimensions: (height, width) => [width, height],
    forward: (_height, width, y, x) => [width - 1 - x, y],
    inverse: (_height, width, y, x) => [x, width - 1 - y],
  },
  {
    id: "transpose",
    dimensions: (height, width) => [width, height],
    forward: (_height, _width, y, x) => [x, y],
    inverse: (_height, _width, y, x) => [x, y],
  },
  {
    id: "transpose-anti",
    dimensions: (height, width) => [width, height],
    forward: (height, width, y, x) => [width - 1 - x, height - 1 - y],
    inverse: (height, width, y, x) => [height - 1 - x, width - 1 - y],
  },
];

function transformPlanMove(move, transform, height, width, inverse = false) {
  const [dy, dx] = DIRS[move];
  const map = inverse ? transform.inverse : transform.forward;
  const [originY, originX] = map(height, width, 1, 1);
  const [nextY, nextX] = map(height, width, 1 + dy, 1 + dx);
  const transformedDy = nextY - originY, transformedDx = nextX - originX;
  return DIRECTION_ENTRIES.find(([, delta]) =>
    delta[0] === transformedDy && delta[1] === transformedDx)?.[0];
}

function canonicalPlanTransform(state) {
  const height = state.rows.length;
  const width = Math.max(...state.rows.map(row => row.length));
  const grid = state.rows.map(row => [...row.padEnd(width, "O")]);
  let staticRobot = null;
  grid.forEach((row, y) => row.forEach((cell, x) => {
    if (cell === "R") staticRobot = [y, x];
  }));
  staticRobot ||= state.robot;
  const candidates = PLAN_BOARD_TRANSFORMS.map(transform => {
    const [nextHeight, nextWidth] = transform.dimensions(height, width);
    const nextGrid = Array.from(
      {length: nextHeight},
      () => Array(nextWidth).fill("O"),
    );
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const [nextY, nextX] = transform.forward(height, width, y, x);
        nextGrid[nextY][nextX] = grid[y][x];
      }
    }
    const rows = nextGrid.map(row => row.join(""));
    const robot = transform.forward(height, width, state.robot[0], state.robot[1]);
    const boxes = state.boxes.map(([position, label]) => {
      const [y, x] = position.split(",").map(Number);
      const [nextY, nextX] = transform.forward(height, width, y, x);
      return {position: pkey(nextY, nextX), label, y: nextY, x: nextX};
    })
      .sort((left, right) =>
        left.y - right.y || left.x - right.x || left.label.localeCompare(right.label))
      .map(({position, label}) => [position, label]);
    const anchor = transform.forward(height, width, staticRobot[0], staticRobot[1]);
    const stateKey = `${rows.join("\n")}|${pkey(robot[0], robot[1])}|` +
      boxes.map(([position, label]) => `${position},${label}`).sort().join(";");
    return {
      transform,
      height,
      width,
      rows,
      robot,
      boxes,
      anchorY: anchor[0],
      anchorHeight: nextHeight,
      stateKey,
    };
  });
  candidates.sort((left, right) => {
    const leftDepth = left.anchorY * Math.max(1, right.anchorHeight - 1);
    const rightDepth = right.anchorY * Math.max(1, left.anchorHeight - 1);
    return rightDepth - leftDepth ||
      left.stateKey.localeCompare(right.stateKey) ||
      left.transform.id.localeCompare(right.transform.id);
  });
  return candidates[0];
}

function restorePlanCheckpoint(checkpoint, canonical, originalRows) {
  if (!checkpoint) return checkpoint;
  const restorePosition = position => {
    const [y, x] = position.split(",").map(Number);
    return pkey(...canonical.transform.inverse(
      canonical.height,
      canonical.width,
      y,
      x,
    ));
  };
  return {
    ...checkpoint,
    state: checkpoint.state && {
      ...checkpoint.state,
      rows: originalRows,
      robot: canonical.transform.inverse(
        canonical.height,
        canonical.width,
        checkpoint.state.robot[0],
        checkpoint.state.robot[1],
      ),
      boxes: checkpoint.state.boxes.map(([position, label]) => [
        restorePosition(position),
        label,
      ]),
    },
    path: checkpoint.path?.map(move => transformPlanMove(
      move,
      canonical.transform,
      canonical.height,
      canonical.width,
      true,
    )),
  };
}

function canonicalPlanMacroBeamSearch(payload) {
  if (payload.planCanonicalOrientation === false) return planMacroBeamSearch(payload);
  const canonical = canonicalPlanTransform(payload.state);
  if (canonical.transform.id === "identity") {
    return {...planMacroBeamSearch(payload), planOrientation: "identity"};
  }
  const result = planMacroBeamSearch({
    ...payload,
    preparedBoard: undefined,
    trackedSignatures: undefined,
    state: {
      ...payload.state,
      rows: canonical.rows,
      robot: canonical.robot,
      boxes: canonical.boxes,
    },
  });
  const restorePath = path => path?.map(move => transformPlanMove(
    move,
    canonical.transform,
    canonical.height,
    canonical.width,
    true,
  ));
  return {
    ...result,
    path: restorePath(result.path),
    checkpoint: restorePlanCheckpoint(result.checkpoint, canonical, payload.state.rows),
    checkpoints: result.checkpoints?.map(checkpoint =>
      restorePlanCheckpoint(checkpoint, canonical, payload.state.rows)),
    planOrientation: canonical.transform.id,
  };
}

function takeDiverse(candidates, count, selected, scoreKey, groupKey = "pushClass") {
  const groups = new Map();
  for (const candidate of candidates) {
    const identity = candidate.exactIdentity ?? candidate.exactSignature;
    if (selected.has(identity)) continue;
    const key = candidate[groupKey] || candidate.pushClass || identity;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(candidate);
  }
  let queues = [...groups.values()].map(items => ({
    items: items.sort((left, right) => left[scoreKey] - right[scoreKey]),
    index: 0,
  }));
  queues.sort((left, right) => left.items[0][scoreKey] - right.items[0][scoreKey]);
  const result = [];
  while (result.length < count && queues.length) {
    const remaining = [];
    for (const queue of queues) {
      if (result.length >= count) break;
      const candidate = queue.items[queue.index++];
      const identity = candidate.exactIdentity ?? candidate.exactSignature;
      if (!selected.has(identity)) {
        selected.add(identity);
        result.push(candidate);
      }
      if (queue.index < queue.items.length) remaining.push(queue);
    }
    queues = remaining;
  }
  return result;
}

function thresholdBucket(value, thresholds) {
  for (let index = 0; index < thresholds.length; index++) {
    if (value <= thresholds[index]) return index;
  }
  return thresholds.length;
}

function centeredFeatureBucket(value) {
  if (value <= -1) return 0;
  if (value < -0.1) return 1;
  if (value <= 0.1) return 2;
  if (value < 1) return 3;
  return 4;
}

function beamFeatureClass(candidate, bestEstimate = candidate.estimate) {
  const slack = candidate.estimate - bestEstimate;
  const mobility = candidate.reachable?.size ?? 0;
  return [
    `h${thresholdBucket(slack, [2, 5, 9])}`,
    `r${thresholdBucket(candidate.topology ?? 0, [0, 1, 3])}`,
    `e${thresholdBucket(candidate.evacuation ?? 0, [0, 1, 3])}`,
    `p${thresholdBucket(candidate.packing ?? 0, [0, 1, 3])}`,
    `g${thresholdBucket(candidate.doorway ?? 0, [0, 1, 3])}${centeredFeatureBucket(candidate.doorwayDelta ?? 0)}`,
    `d${centeredFeatureBucket(candidate.dependencyDelta ?? 0)}${centeredFeatureBucket(candidate.localRoomDelta ?? 0)}`,
    `m${thresholdBucket(mobility, [0, 8, 20])}`,
  ].join("|");
}

function selectBeamLayer(candidates, width, profile = "balanced", metrics = null,
  useFeatureSpace = true) {
  if (candidates.length <= width) return candidates;
  let bestEstimate = Infinity;
  for (const candidate of candidates) bestEstimate = Math.min(bestEstimate, candidate.estimate);
  const bands = [[], [], [], []];
  for (const candidate of candidates) {
    const slack = candidate.estimate - bestEstimate;
    bands[slack <= 2 ? 0 : slack <= 5 ? 1 : slack <= 9 ? 2 : 3].push(candidate);
  }
  const ratios = profile === "milestone"
    ? [0.20, 0.20, 0.20, 0.40]
    : profile === "detour"
    ? [0.30, 0.25, 0.25, 0.20]
    : [0.50, 0.25, 0.15, 0.10];
  const groupKey = profile === "milestone" ? "strategicClass" : "pushClass";
  const selected = new Set(), result = [];
  let featureSelectedCount = 0;
  if (useFeatureSpace) {
    const featureRatio = profile === "milestone" ? 0.55 : profile === "detour" ? 0.45 : 0.35;
    for (const candidate of candidates) {
      candidate.featureClass = beamFeatureClass(candidate, bestEstimate);
      candidate.featureArchiveScore = Number.isFinite(candidate.exploreScore)
        ? candidate.exploreScore
        : candidate.score;
    }
    const cells = new Set(candidates.map(candidate => candidate.featureClass));
    const featureQuota = Math.max(1, Math.floor(width * featureRatio));
    const featureSelected = takeDiverse(
      candidates, featureQuota, selected, "featureArchiveScore", "featureClass");
    result.push(...featureSelected);
    featureSelectedCount = featureSelected.length;
    if (metrics) {
      metrics.beamFeatureCells += cells.size;
      metrics.beamFeatureSelections += featureSelected.length;
    }
  }
  const bandWidth = width - result.length;
  bands.forEach((band, index) => {
    const quota = index === bands.length - 1
      ? bandWidth - ratios.slice(0, index)
        .reduce((total, ratio) => total + Math.floor(bandWidth * ratio), 0)
      : Math.floor(bandWidth * ratios[index]);
    const scoreKey = index === bands.length - 1 ? "exploreScore" : "score";
    result.push(...takeDiverse(band, quota, selected, scoreKey, groupKey));
  });
  if (result.length < width) {
    const ranked = [...candidates].sort((left, right) => left.score - right.score);
    result.push(...takeDiverse(ranked, width - result.length, selected, "score", groupKey));
  }
  if (metrics) metrics.beamBandSelections += result.length - featureSelectedCount;
  return result;
}

function planMilestoneSignature(candidate, board) {
  const solved = candidate.boxes
    .filter(([y, x, label]) => board.goals.get(pkey(y, x)) === label)
    .map(([y, x]) => pkey(y, x))
    .sort()
    .join(".");
  const blocked = candidate.goalAccess.blockedGoals
    .map(goal => goal.goal)
    .sort()
    .join(".");
  const schedule = candidate.doorwaySchedule
    ? `${candidate.doorwaySchedule.pendingExports}.` +
      `${candidate.doorwaySchedule.remainingImports}.` +
      `${candidate.doorwaySchedule.unpackedImports}.` +
      `${candidate.doorwaySchedule.prematureImports}.` +
      `${candidate.doorwaySchedule.crossingConflicts}.` +
      `${candidate.doorwaySchedule.stagingBlockers}.` +
      `${candidate.doorwaySchedule.blockedImportAccess}.` +
      `${candidate.doorwaySchedule.packingOrderViolations}`
    : "";
  return `${solved}|${blocked}|${schedule}|${roomFlowSignature(candidate.boxes, board)}`;
}

function selectPlanLayer(candidates, width, board) {
  const ranked = [...candidates].sort((left, right) =>
    left.score - right.score ||
    left.estimate - right.estimate ||
    left.cost - right.cost);
  if (ranked.length <= width) return ranked;
  const groups = new Map();
  for (const candidate of ranked) {
    const key = planMilestoneSignature(candidate, board);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(candidate);
  }
  const selected = [], selectedIds = new Set();
  const structuralEliteCount = Math.max(1, Math.ceil(width * 0.1));
  const structuralElites = [...ranked]
    .filter(candidate => Number.isFinite(candidate.planCheckpointRank))
    .sort((left, right) =>
      left.planCheckpointRank - right.planCheckpointRank ||
      left.score - right.score)
    .slice(0, structuralEliteCount);
  for (const candidate of structuralElites) {
    selected.push(candidate);
    selectedIds.add(candidate.exactIdentity);
  }
  const heuristicEliteCount = Math.max(1, Math.ceil(width * 0.1));
  const heuristicElites = [...ranked]
    .sort((left, right) =>
      left.estimate - right.estimate ||
      left.cost - right.cost ||
      left.score - right.score)
    .slice(0, heuristicEliteCount);
  for (const candidate of heuristicElites) {
    if (selectedIds.has(candidate.exactIdentity)) continue;
    selected.push(candidate);
    selectedIds.add(candidate.exactIdentity);
  }
  let queues = [...groups.values()];
  while (selected.length < Math.ceil(width * 0.7) && queues.length) {
    const remaining = [];
    for (const queue of queues) {
      if (selected.length >= Math.ceil(width * 0.7)) break;
      const candidate = queue.shift();
      if (!selectedIds.has(candidate.exactIdentity)) {
        selected.push(candidate);
        selectedIds.add(candidate.exactIdentity);
      }
      if (queue.length) remaining.push(queue);
    }
    queues = remaining;
  }
  for (const candidate of ranked) {
    if (selected.length >= width) break;
    if (selectedIds.has(candidate.exactIdentity)) continue;
    selected.push(candidate);
    selectedIds.add(candidate.exactIdentity);
  }
  return selected;
}

function planMacroBeamSearch(payload) {
  const board = payload.preparedBoard || parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
    cost: 0,
    node: null,
  };
  const width = payload.planBeamWidth || payload.beamWidth || 80;
  const maxSegments = payload.maxPlanSegments || 80;
  const maxVisited = payload.maxVisited || 20000;
  const maxPushes = payload.maxDepth || 320;
  const boxBranchLimit = payload.planBoxBranches || 8;
  const macroLimit = payload.sequenceMacroLimit || 24;
  const macroExplored = payload.sequenceMacroExplored || 64;
  const macroResults = payload.sequenceMacroResults || 5;
  const seen = new BoundedDepthMap(payload.transpositionLimit || 60000);
  const seenExact = new BoundedDepthMap(payload.transpositionLimit || 60000);
  let beam = [initial], visited = 0, generated = 0, peakFrontier = 1;
  let trackedThrough = payload.trackedSignatures ? 0 : undefined;
  let bestEstimate = discoveryHeuristic(initial.boxes, board), bestPushes = 0;
  const planBound = Math.min(
    Number.isFinite(payload.upperBound) ? payload.upperBound : Infinity,
    bestEstimate + (payload.planSlack ?? 192),
  );
  let bestCheckpoint = null, bestCheckpointRank = Infinity;
  let bestHeuristicCheckpoint = null;
  const rootDoorwayTasks = payload.planDoorwaySchedule === false
    ? [] : assignmentDoorwayPlan(initial.boxes, board, true).tasks;
  const hasEvacuationPlan = rootDoorwayTasks.some(task => task.direction === "export");
  const evaluateDoorwaySchedule = boxes =>
    doorwayScheduleState(boxes, board, rootDoorwayTasks);
  const importAccessBlockers = (state, reachable) => {
    const blockers = new Map();
    if (!state.doorwaySchedule.blockedImportAccess) return blockers;
    const routes = minimumBlockerRoutes(reachable, board);
    const occupied = new Set(state.boxes.map(([y, x]) => pkey(y, x)));
    for (const task of rootDoorwayTasks.filter(task => task.direction === "import")) {
      const room = board.topology.rooms[task.roomIndex];
      const [y, x, label] = state.boxes[task.boxIndex];
      const position = pkey(y, x);
      if (room.cells.has(position)) continue;
      const currentDistance = playerAwarePushDistances(board, position).get(room.gate);
      const options = [];
      for (const [, [dy, dx]] of DIRECTION_ENTRIES) {
        const destination = pkey(y + dy, x + dx);
        const support = pkey(y - dy, x - dx);
        if (!board.floor.has(destination) || occupied.has(destination) ||
            staticDead(y + dy, x + dx, board, label)) continue;
        const nextDistance = playerAwarePushDistances(board, destination).get(room.gate);
        if (nextDistance >= currentDistance) continue;
        const candidateBoxes = state.boxes.slice();
        candidateBoxes[task.boxIndex] = [y + dy, x + dx, label];
        if (createsDynamicDeadlock(candidateBoxes, board, [y + dy, x + dx])) {
          state.boxes.forEach(([boxY, boxX], index) => {
            if (index === task.boxIndex) return;
            if (Math.abs(boxY - (y + dy)) <= 1 && Math.abs(boxX - (x + dx)) <= 1) {
              const blocker = pkey(boxY, boxX);
              blockers.set(blocker, (blockers.get(blocker) || 0) + 1);
            }
          });
          continue;
        }
        const route = routes.routeTo(board.dense.idByKey.get(support) ?? -1);
        if (route) options.push(route);
      }
      const minimum = Math.min(...options.map(option => option.blockerCount));
      for (const option of options.filter(candidate => candidate.blockerCount === minimum)) {
        for (const blocker of option.blockers) {
          blockers.set(blocker, (blockers.get(blocker) || 0) + 1);
        }
      }
    }
    return blockers;
  };
  const scoreCandidate = child => {
    child.estimate = discoveryHeuristic(child.boxes, board);
    child.goalAccess = child.macroContext?.goalAccess ||
      goalAccessAnalysis(child.boxes, board);
    child.evacuation = roomEvacuationPenalty(child.boxes, board);
    child.doorwaySchedule = child.macroContext?.doorwaySchedule ||
      evaluateDoorwaySchedule(child.boxes);
    const evacuationActive = child.doorwaySchedule.pendingExports > 0 ||
      child.doorwaySchedule.stagingBlockers > 0 ||
      child.doorwaySchedule.blockedImportAccess > 0;
    const evacuationComplete = hasEvacuationPlan &&
      child.doorwaySchedule.pendingExports === 0;
    child.score = child.cost + (evacuationActive ? 0.25 : 1.15) * child.estimate +
      4 * child.goalAccess.penalty + 0.08 * child.evacuation +
      (evacuationActive ? 4 : 3) * child.doorwaySchedule.penalty -
      (evacuationComplete ? 250 : 0);
    return child;
  };
  const checkpointRank = child => {
    const schedule = child.doorwaySchedule;
    const unsafe = schedule.prematureImports + schedule.gateBlockers +
      schedule.crossingConflicts + schedule.strandedExports +
      schedule.packingOrderViolations;
    const remaining = schedule.pendingExports + schedule.remainingImports +
      schedule.unpackedImports;
    const evacuationComplete = hasEvacuationPlan && schedule.pendingExports === 0;
    return 1000 * unsafe + 100 * remaining -
      (evacuationComplete ? 200 : 0) +
      50 * schedule.blockedImportAccess + 20 * schedule.stagingBlockers +
      2 * child.cost + child.estimate + 4 * child.goalAccess.penalty;
  };
  initial.exactIdentity = exactPushIdentity(initial, board);
  initial.goalAccess = goalAccessAnalysis(initial.boxes, board);
  initial.doorwaySchedule = evaluateDoorwaySchedule(initial.boxes);
  seenExact.set(initial.exactIdentity, 0);

  for (let segment = 0;
    segment < maxSegments && beam.length && visited < maxVisited;
    segment++) {
    const candidates = new Map();
    for (const current of beam) {
      if (visited++ >= maxVisited) break;
      const reachable = reachablePaths(current, board);
      if (createsSealedCorralDeadlock(current, board, reachable)) continue;
      const accessBlockers = importAccessBlockers(current, reachable);
      const firstPushes = pushNeighbors(current, board, reachable);
      const rankedFirst = firstPushes.map(next => {
        const estimate = discoveryHeuristic(next.boxes, board);
        const accessDelta = goalAccessDelta(current.goalAccess, current, next, board);
        const evacuation = roomEvacuationPenalty(next.boxes, board);
        const schedule = evaluateDoorwaySchedule(next.boxes);
        const blockerProgress = accessBlockers.get(next.pushedFrom) || 0;
        const estimateWeight = current.doorwaySchedule.pendingExports ||
          current.doorwaySchedule.stagingBlockers ||
          current.doorwaySchedule.blockedImportAccess ? 0.25 : 1;
        const completesEvacuation = hasEvacuationPlan &&
          current.doorwaySchedule.pendingExports > 0 &&
          schedule.pendingExports === 0;
        return {
          next,
          score: estimateWeight * estimate + 5 * accessDelta + 0.08 * evacuation +
            4 * (schedule.penalty - current.doorwaySchedule.penalty) -
            (completesEvacuation ? 250 : 0) - 12 * blockerProgress,
        };
      }).sort((left, right) => left.score - right.score);
      const selectedBoxes = new Set(), selectedFirst = [];
      for (const candidate of rankedFirst) {
        if (selectedBoxes.has(candidate.next.pushedFrom)) continue;
        selectedBoxes.add(candidate.next.pushedFrom);
        selectedFirst.push(candidate.next);
        if (selectedBoxes.size >= boxBranchLimit) break;
      }
      for (const candidate of rankedFirst) {
        if (selectedFirst.length >= boxBranchLimit + 2) break;
        if (selectedFirst.includes(candidate.next)) continue;
        selectedFirst.push(candidate.next);
      }
      for (let firstIndex = 0; firstIndex < selectedFirst.length; firstIndex++) {
        const first = selectedFirst[firstIndex];
        const movedIndex = current.boxes.findIndex(([y, x]) =>
          pkey(y, x) === first.pushedFrom);
        const doorwayTask = rootDoorwayTasks.find(task => task.boxIndex === movedIndex);
        const doorwayRoom = doorwayTask
          ? board.topology.rooms[doorwayTask.roomIndex] : null;
        const currentPosition = movedIndex >= 0
          ? pkey(current.boxes[movedIndex][0], current.boxes[movedIndex][1]) : null;
        const crossingComplete = doorwayTask?.direction === "export"
          ? !doorwayRoom.cells.has(currentPosition) && currentPosition !== doorwayRoom.gate
          : doorwayTask?.direction === "import"
            ? doorwayRoom.cells.has(currentPosition)
            : false;
        const clearingStaging = doorwayTask &&
          current.doorwaySchedule.stagingBlockers > 0 &&
          doorwayRoom.exteriorStaging.has(currentPosition);
        const assignedTarget = doorwayTask?.target ||
          cacheDiscoveryAssignmentDetail(
            current.boxes,
            board,
          ).assignedTargets.get(movedIndex);
        const objective = clearingStaging
          ? {direction: "clear", roomIndex: doorwayTask.roomIndex}
          : doorwayTask && !crossingComplete
            ? doorwayTask : assignedTarget ? {target: assignedTarget} : null;
        const sameBoxDirections = rankedFirst.filter(candidate =>
          candidate.next.pushedFrom === first.pushedFrom).length;
        const ambiguity = sameBoxDirections +
          Number(Boolean(doorwayTask)) +
          Number(current.doorwaySchedule.crossingConflicts > 0) +
          Number(current.doorwaySchedule.stagingBlockers > 0);
        const forcedMacro = current.boxes.length <= 4 &&
          sameBoxDirections === 1 &&
          board.topology.tunnels.has(first.pushedTo);
        const cheapExplored = ambiguity <= 2 ? 16 : 32;
        const cheapResults = macroResults;
        const fullExplored = objective
          ? payload.targetedMacroExplored || Math.max(96, macroExplored)
          : macroExplored;
        const movedPosition = sequence =>
          pkey(sequence.boxes[movedIndex][0], sequence.boxes[movedIndex][1]);
        const intermediateGuard = sequence => {
          if (!doorwayTask && !assignedTarget) return false;
          const position = movedPosition(sequence);
          const crossedDoorway = doorwayTask?.direction === "export"
            ? (doorwayRoom.cells.has(currentPosition) ||
                currentPosition === doorwayRoom.gate) &&
              !doorwayRoom.cells.has(position) && position !== doorwayRoom.gate
            : doorwayTask?.direction === "import" &&
              !doorwayRoom.cells.has(currentPosition) &&
              doorwayRoom.cells.has(position);
          const objectiveComplete = position === assignedTarget ||
            crossedDoorway ||
            (clearingStaging && !doorwayRoom.exteriorStaging.has(position) &&
              position !== doorwayRoom.gate);
          if (!objectiveComplete) return false;
          const doorwaySchedule = evaluateDoorwaySchedule(sequence.boxes);
          const goalAccess = goalAccessAnalysis(sequence.boxes, board);
          sequence.macroContext = {doorwaySchedule, goalAccess};
          if (payload.planEgressGuard !== false &&
              doorwayTask?.direction === "import" &&
              doorwaySchedule.strandedExports >
                current.doorwaySchedule.strandedExports) return "stranded-export";
          return false;
        };
        const expand = (explored, results) => objective
          ? expandTargetedPushSequence(
            first, board, objective, macroLimit, explored, results,
            {lockProven: false, intermediateGuard:
              payload.incrementalMacroGuard === false ? undefined : intermediateGuard},
          )
          : expandPushSequences(
            first, board, macroLimit, explored, results,
            {lockProven: false, intermediateGuard:
              payload.incrementalMacroGuard === false ? undefined : intermediateGuard},
          );
        let expanded;
        if (payload.adaptiveMacroEffort === false) {
          if (activePerformance) activePerformance.macroFullExpansions++;
          expanded = expand(fullExplored, macroResults);
        } else if (!forcedMacro) {
          if (activePerformance) activePerformance.macroFullExpansions++;
          expanded = expand(fullExplored, macroResults);
        } else {
          if (activePerformance) activePerformance.macroCheapExpansions++;
          expanded = expand(Math.min(cheapExplored, fullExplored), cheapResults);
          const cheapEndpoints = expanded.filter(next => next.pushes > 1);
          if (fullExplored > cheapExplored &&
              cheapEndpoints.length === 0) {
            if (activePerformance) {
              activePerformance.macroWidenings++;
              activePerformance.macroFullExpansions++;
            }
            expanded = expand(fullExplored, macroResults);
          }
        }
        const endpoints = expanded.filter(next => next.pushes > 1);
        const successors = endpoints.length
          ? (firstIndex < 2 ? [expanded[0], ...endpoints] : endpoints)
          : expanded;
        for (const next of successors) {
          const cost = current.cost + next.pushes;
          if (cost > maxPushes) continue;
          const child = scoreCandidate({
            robot: next.robot,
            boxes: next.boxes,
            cost,
            node: {parent: current.node, segment: next.path},
            pushClass: next.pushClass,
            macroContext: next.macroContext,
          });
          if (payload.planEgressGuard !== false &&
              doorwayTask?.direction === "import" &&
              child.doorwaySchedule.strandedExports >
                current.doorwaySchedule.strandedExports) continue;
          if (payload.planEgressGuard !== false &&
              child.doorwaySchedule.packingOrderViolations >
                current.doorwaySchedule.packingOrderViolations) continue;
          if (payload.planGoalAccessGuard !== false) {
            const blockedBefore = new Set(
              current.goalAccess.blockedGoals.map(goalState => goalState.goal),
            );
            if (child.goalAccess.blockedGoals.some(goalState =>
              !blockedBefore.has(goalState.goal))) continue;
          }
          if (payload.planEgressGuard !== false && assignedTarget) {
            const [movedY, movedX] = child.boxes[movedIndex];
            const movedPosition = pkey(movedY, movedX);
            if (movedPosition !== assignedTarget) {
              const crossedDoorway = doorwayTask?.direction === "export"
                ? (doorwayRoom.cells.has(currentPosition) ||
                    currentPosition === doorwayRoom.gate) &&
                  !doorwayRoom.cells.has(movedPosition) &&
                  movedPosition !== doorwayRoom.gate
                : doorwayTask?.direction === "import" &&
                  !doorwayRoom.cells.has(currentPosition) &&
                  doorwayRoom.cells.has(movedPosition);
              const childReachable = reachablePaths(child, board);
              if (!crossedDoorway && !pushBoxNeighbors(
                child,
                board,
                movedPosition,
                childReachable,
                {lockProven: false},
              ).length) continue;
            }
          }
          if (!Number.isFinite(child.estimate)) continue;
          if (child.cost + child.estimate > planBound) continue;
          generated++;
          if (goal(child.boxes, board.goals)) {
            return {
              path: reconstructNodePath(child.node),
              visited,
              generated,
              retained: seen.size + seenExact.size,
              peakFrontier,
              bestEstimate: 0,
              bestPushes: cost,
              strategy: "Plan Macro Beam",
            };
          }
          child.exactIdentity = exactPushIdentity(child, board);
          if ((seenExact.get(child.exactIdentity) ?? Infinity) <= cost) continue;
          const existing = candidates.get(child.exactIdentity);
          if (!existing || child.score < existing.score) {
            candidates.set(child.exactIdentity, child);
          }
          if (child.estimate < bestEstimate ||
              (child.estimate === bestEstimate && cost < bestPushes)) {
            bestEstimate = child.estimate;
            bestPushes = cost;
            bestHeuristicCheckpoint = child;
          }
          const childCheckpointRank = checkpointRank(child);
          child.planCheckpointRank = childCheckpointRank;
          if (childCheckpointRank < bestCheckpointRank) {
            bestCheckpointRank = childCheckpointRank;
            bestCheckpoint = child;
          }
        }
      }
    }
    peakFrontier = Math.max(peakFrontier, candidates.size);
    const eligible = [];
    for (const child of selectPlanLayer(candidates.values(), width * 2, board)) {
      const reachable = reachablePaths(child, board);
      if (createsSealedCorralDeadlock(child, board, reachable)) continue;
      child.identity = pushIdentity(child, reachable);
      if ((seen.get(child.identity) ?? Infinity) <= child.cost) continue;
      child.signature = pushKey(child, reachable);
      eligible.push(child);
    }
    beam = selectPlanLayer(eligible, width, board);
    for (const child of beam) {
      seen.set(child.identity, child.cost);
      seenExact.set(child.exactIdentity, child.cost);
      if (payload.trackedSignatures?.[child.cost] === child.signature) {
        trackedThrough = Math.max(trackedThrough, child.cost);
      }
    }
    if ((segment + 1) % 2 === 0) {
      postMessage({
        type: "progress",
        visited,
        bestEstimate,
        bestPushes,
        depth: segment + 1,
        frontier: beam.length,
        generated,
        performance: performanceSnapshot(board.metrics),
      });
    }
  }
  const checkpoint = serializeSearchCheckpoint(bestCheckpoint, board);
  const heuristicCheckpoint = serializeSearchCheckpoint(bestHeuristicCheckpoint, board);
  const checkpoints = [checkpoint];
  if (heuristicCheckpoint &&
      (!checkpoint || heuristicCheckpoint.cost !== checkpoint.cost ||
        heuristicCheckpoint.estimate !== checkpoint.estimate)) {
    checkpoints.push(heuristicCheckpoint);
  }
  return {
    path: null,
    visited,
    generated,
    retained: seen.size + seenExact.size,
    peakFrontier,
    bestEstimate,
    bestPushes,
    trackedThrough,
    checkpoint,
    checkpoints: checkpoints.filter(Boolean),
    cutoff: true,
    terminationReason: visited >= maxVisited ? "state-budget" : "plan-frontier-exhausted",
  };
}

function beamSearch(payload) {
  const board = payload.preparedBoard || parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
    cost: 0,
  };
  const width = payload.beamWidth || 3000;
  const maxDepth = payload.maxDepth || 500;
  const weight = payload.weight || 3;
  const diversity = payload.diversity ?? 1.5;
  const goalPackingWeight = payload.goalPackingWeight ?? 0.8;
  const mobilityWeight = payload.mobilityWeight ?? 0.03;
  const topologyWeight = payload.topologyWeight ?? 0.7;
  const evacuationWeight = payload.evacuationWeight ?? 0;
  const supportDependencyWeight = payload.supportDependencyWeight ?? 0.8;
  const localRoomWeight = payload.localRoomWeight ?? 0.6;
  const doorwayFlowWeight = payload.doorwayFlowWeight ?? 0.35;
  const lockProvenCommitments = payload.lockProvenCommitments !== false;
  const beamProfile = payload.beamProfile || "balanced";
  const seed = payload.seed || 0;
  const transpositionLimit = payload.transpositionLimit || Math.max(12000, width * 60);
  const seenDepth = new BoundedDepthMap(transpositionLimit);
  const seenExactDepth = new BoundedDepthMap(Math.max(8000, Math.floor(transpositionLimit / 2)));
  const handoffLimit = payload.checkpointLimit || 12;
  const progressInterval = payload.progressInterval || 5000;
  const progressIntervalMs = payload.progressIntervalMs || 5000;
  const handoffCheckpoints = new Map();
  let visited = 0, reported = 0, bestEstimate = Infinity, bestPushes = 0;
  let generated = 0, peakFrontier = 1;
  let lastProgressAt = now();
  let beamCutoff = false;
  let bestCheckpoint = null;
  let bestHandoff = null;
  let phaseHandoff = null;
  const endgameCheckpoints = [];
  let trackedThrough = payload.trackedSignatures ? 0 : undefined;
  const childDoorwayGate = createOrderingProductivityGate(
    payload.strategicSignalWarmup || 64,
    payload.strategicSignalCooldown || 512,
  );
  const childPackingGate = createOrderingProductivityGate(
    payload.strategicSignalWarmup || 64,
    payload.strategicSignalCooldown || 512,
  );

  initial.reachable = reachablePaths(initial, board);
  if (createsSealedCorralDeadlock(initial, board, initial.reachable)) {
    return {path: null, visited};
  }
  initial.identity = pushIdentity(initial, initial.reachable);
  initial.signature = pushKey(initial, initial.reachable);
  initial.strategicHistory = "";
  initial.openingHistory = "";
  const initialEstimate = heuristic(initial.boxes, board);
  if (!Number.isFinite(initialEstimate)) return {path: null, visited};
  bestEstimate = initialEstimate;
  seenDepth.set(initial.identity, 0);
  let beam = [initial];

  searchLayers: for (let depth = 0; beam.length && depth <= maxDepth; depth++) {
    const candidates = new Map();
    for (const current of beam) {
      visited++;
      if (goal(current.boxes, board.goals)) {
        return {
          path: reconstructNodePath(current.node),
          visited,
          generated,
          retained: seenDepth.size,
          peakFrontier,
          transpositionEvictions: seenDepth.evictions + seenExactDepth.evictions,
        };
      }
      if (payload.maxVisited && visited >= payload.maxVisited) {
        beamCutoff = true;
        break searchLayers;
      }
      const dependencyGraph = supportDependencyGraph(current, board, current.reachable);
      const localRooms = [
        ...exactLocalRoomAnalyses(current, board, current.reachable),
        ...exactLocalCorralAnalyses(current, board, current.reachable),
      ];
      const doorwayBefore = typedDoorwayFlow(current.boxes, board);
      const goalAccessBefore = payload.goalAccessOrdering === false
        ? null : goalAccessAnalysis(current.boxes, board);
      const currentCommitments = lockProvenCommitments ? goalCommitments(current.boxes, board, {
        doorway: doorwayBefore,
        supportDependency: dependencyGraph,
        localAnalyses: localRooms,
      }) : null;
      for (const rawNext of pushNeighbors(
        current,
        board,
        current.reachable,
        {commitments: currentCommitments},
      )) {
        const expansions = payload.straightMacros
          ? expandStraightPushes(
              rawNext,
              board,
              payload.straightMacroLimit || 8,
              {lockProven: lockProvenCommitments},
            )
          : payload.sequenceMacros
          ? expandPushSequences(
              rawNext,
              board,
              payload.sequenceMacroLimit || 12,
              payload.sequenceMacroExplored || 48,
              payload.sequenceMacroResults || 8,
              {lockProven: lockProvenCommitments},
            )
          : [expandPushMacro(
              rawNext,
              board,
              payload.forcedMacros !== false,
              {lockProven: lockProvenCommitments},
            )].filter(Boolean);
        for (const next of expansions) {
        const child = {robot: next.robot, boxes: next.boxes, cost: current.cost + next.pushes};
        if (child.cost > maxDepth) continue;
        if (payload.upperBound && child.cost > payload.upperBound) continue;
        if (goal(child.boxes, board.goals)) {
          return {
            path: [...reconstructNodePath(current.node), ...next.path],
            visited,
            generated: generated + candidates.size + 1,
            retained: seenDepth.size,
            peakFrontier: Math.max(peakFrontier, beam.length, candidates.size + 1),
            transpositionEvictions: seenDepth.evictions + seenExactDepth.evictions,
          };
        }
        child.exactIdentity = exactPushIdentity(child, board);
        if ((seenExactDepth.get(child.exactIdentity) ?? Infinity) <= child.cost) continue;
        const estimate = heuristic(child.boxes, board);
        if (!Number.isFinite(estimate)) continue;
        if (payload.upperBound && child.cost + estimate > payload.upperBound) continue;
        const previousBestEstimate = bestEstimate;
        if (estimate < bestEstimate) {
          bestEstimate = estimate;
          bestPushes = child.cost;
        }
        const topology = topologyPenalty(child.boxes, board);
        const dependencyDelta = supportDependencyDelta(dependencyGraph, next);
        const localRoomDelta = localRoomOrderingDelta(localRooms, next);
        const evaluateDoorway = childDoorwayGate.shouldEvaluate();
        const evaluatePacking = childPackingGate.shouldEvaluate();
        const doorway = evaluateDoorway || evaluatePacking
          ? typedDoorwayFlow(child.boxes, board)
          : doorwayBefore;
        const packing = evaluatePacking ? goalPackingBonus(child.boxes, board, {
          doorway,
          supportDependency: dependencyGraph,
          localAnalyses: localRooms,
          transition: next,
        }) : 0;
        const usefulSignal = estimate < previousBestEstimate ||
          goal(child.boxes, board.goals);
        for (const [gate, evaluated] of [
          [childDoorwayGate, evaluateDoorway],
          [childPackingGate, evaluatePacking],
        ]) {
          if (evaluated) {
            board.metrics.strategicSignalEvaluations++;
            gate.observe({changed: true, useful: usefulSignal});
            if (usefulSignal) board.metrics.strategicSignalUseful++;
          } else {
            board.metrics.strategicSignalSkips++;
          }
        }
        const doorwayDelta = doorwayFlowDelta(doorwayBefore, current, next);
        const relevance = relevanceOrderingScore(current, board, next, {
          supportDependency: dependencyGraph,
          doorway: doorwayBefore,
          goalAccess: goalAccessBefore,
          recentPush: current.recentPush,
        });
        const relevanceScore = recordRelevanceOrdering(board.metrics, relevance);
        const evacuation = evacuationWeight
          ? roomEvacuationPenalty(child.boxes, board)
          : 0;
        if (beamProfile === "milestone") {
          const transition = roomTransitionEvent(current.boxes, child.boxes, board);
          child.strategicHistory = transition
            ? `${current.strategicHistory || ""}>${transition}`.split(">").slice(-4).join(">")
            : current.strategicHistory || "";
          child.openingHistory = child.cost <= 10
            ? `${current.openingHistory || ""}/${next.pushClass}`
            : current.openingHistory || "";
        }
        const score = (payload.costWeight || 0) * child.cost +
          weight * estimate + topologyWeight * topology +
          evacuationWeight * evacuation -
          goalPackingWeight * packing +
          supportDependencyWeight * dependencyDelta +
          localRoomWeight * localRoomDelta +
          doorwayFlowWeight *
            (0.2 * (evaluateDoorway ? doorway.penalty : doorwayBefore.penalty) +
              doorwayDelta) +
          (payload.relevanceWeight ?? 0.6) * relevanceScore +
          diversity * signatureNoise(child.exactIdentity, seed);
        const exploreScore = topologyWeight * topology + evacuationWeight * evacuation -
          goalPackingWeight * packing +
          supportDependencyWeight * dependencyDelta +
          localRoomWeight * localRoomDelta +
          doorwayFlowWeight *
            (0.2 * (evaluateDoorway ? doorway.penalty : doorwayBefore.penalty) +
              doorwayDelta) +
          (payload.relevanceWeight ?? 0.6) * relevanceScore +
          diversity * signatureNoise(child.exactIdentity, seed + 7919);
        const existing = candidates.get(child.exactIdentity);
        if (!existing || score < existing.score) {
          const candidate = {
            ...child,
            node: {parent: current.node || null, segment: next.path},
            estimate,
            topology,
            evacuation,
            packing,
            dependencyDelta,
            relevance: relevance.signals,
            localRoomDelta,
            doorway: doorway.penalty,
            doorwayDelta,
            score,
            exploreScore,
            pushClass: next.pushClass,
            strategicClass: beamProfile === "milestone"
              ? `${child.openingHistory}|${child.strategicHistory}|${roomFlowSignature(child.boxes, board)}`
              : null,
            strategicHistory: child.strategicHistory,
            openingHistory: child.openingHistory,
            recentPush: {pushedFrom: next.pushedFrom, pushedTo: next.pushedTo},
          };
          candidates.set(child.exactIdentity, candidate);
          if (!bestCheckpoint || estimate < bestCheckpoint.estimate ||
              (estimate === bestCheckpoint.estimate && child.cost < bestCheckpoint.cost)) {
            bestCheckpoint = candidate;
          }
          if ((payload.endgameVisited || payload.continuationVisited) &&
              estimate <= (payload.endgameThreshold || 60)) {
            const solvedGoals = candidate.boxes
              .filter(([y, x, label]) => board.goals.get(pkey(y, x)) === label)
              .map(([y, x, label]) => `${y},${x},${label}`)
              .sort()
              .join(";");
            candidate.checkpointClass =
              `${roomFlowSignature(candidate.boxes, board)}|${solvedGoals}|${next.pushClass}`;
            candidate.checkpointBand = Math.floor(estimate / 10);
            const existingCheckpoint = endgameCheckpoints.findIndex(checkpoint =>
              checkpoint.checkpointClass === candidate.checkpointClass);
            if (existingCheckpoint >= 0) {
              if (candidate.estimate >= endgameCheckpoints[existingCheckpoint].estimate) continue;
              endgameCheckpoints.splice(existingCheckpoint, 1);
            }
            endgameCheckpoints.push(candidate);
            endgameCheckpoints.sort((left, right) =>
              left.estimate - right.estimate ||
              (left.cost + left.estimate) - (right.cost + right.estimate) ||
              left.cost - right.cost);
            if (endgameCheckpoints.length > (payload.endgameCandidates || 24)) {
              const bandCounts = new Map();
              endgameCheckpoints.forEach(checkpoint => bandCounts.set(
                checkpoint.checkpointBand,
                (bandCounts.get(checkpoint.checkpointBand) || 0) + 1,
              ));
              let crowdedBand = null, crowdedCount = 0;
              for (const [band, count] of bandCounts) {
                if (count > crowdedCount || (count === crowdedCount && band < crowdedBand)) {
                  crowdedBand = band;
                  crowdedCount = count;
                }
              }
              for (let remove = endgameCheckpoints.length - 1; remove >= 0; remove--) {
                if (endgameCheckpoints[remove].checkpointBand !== crowdedBand) continue;
                endgameCheckpoints.splice(remove, 1);
                break;
              }
            }
          }
        }
        }
      }
      const progressNow = now();
      if (visited - reported >= progressInterval ||
          progressNow - lastProgressAt >= progressIntervalMs) {
        postMessage({type: "progress", visited: (payload.progressOffset || 0) + visited,
          bestEstimate, bestPushes, frontier: beam.length, depth,
          performance: performanceSnapshot(board.metrics)});
        reported = visited;
        lastProgressAt = progressNow;
      }
    }
    generated += candidates.size;
    peakFrontier = Math.max(peakFrontier, beam.length, candidates.size);
    const shortlist = selectBeamLayer(
      [...candidates.values()],
      width * 3,
      beamProfile,
      board.metrics,
      payload.featureSpaceQueues !== false,
    );
    beam = [];
    for (const child of shortlist) {
      child.reachable = reachablePaths(child, board);
      if (createsSealedCorralDeadlock(child, board, child.reachable)) continue;
      child.identity = pushIdentity(child, child.reachable);
      if ((seenDepth.get(child.identity) ?? Infinity) <= child.cost) continue;
      child.signature = pushKey(child, child.reachable);
      child.score -= mobilityWeight * child.reachable.size;
      child.exploreScore -= mobilityWeight * child.reachable.size;
      seenDepth.set(child.identity, child.cost);
      seenExactDepth.set(child.exactIdentity, child.cost);
      beam.push(child);
      if (!bestHandoff || child.estimate < bestHandoff.estimate ||
          (child.estimate === bestHandoff.estimate && child.cost < bestHandoff.cost)) {
        bestHandoff = child;
      }
      if (!handoffCheckpoints.has(child.signature)) {
        handoffCheckpoints.set(child.signature, child);
        if (handoffCheckpoints.size > handoffLimit * 3) {
          const retained = [...handoffCheckpoints.entries()]
            .sort(([, left], [, right]) =>
              left.estimate - right.estimate ||
              (left.cost + left.estimate) - (right.cost + right.estimate))
            .slice(0, handoffLimit);
          handoffCheckpoints.clear();
          retained.forEach(([signature, checkpoint]) =>
            handoffCheckpoints.set(signature, checkpoint));
        }
      }
      if (evacuationWeight && child.evacuation === 0 &&
          (!phaseHandoff || child.cost + child.estimate <
            phaseHandoff.cost + phaseHandoff.estimate)) {
        phaseHandoff = child;
      }
    }
    beam = selectBeamLayer(
      beam,
      width,
      beamProfile,
      board.metrics,
      payload.featureSpaceQueues !== false,
    );
    if (payload.trackedSignatures) {
      for (const child of beam) {
        if (payload.trackedSignatures[child.cost] === child.signature) {
          trackedThrough = Math.max(trackedThrough, child.cost);
        }
      }
    }
  }
  const probeCheckpoints = stratifiedCheckpoints(endgameCheckpoints);
  if (payload.continuationVisited && probeCheckpoints.length) {
    let remainingVisited = payload.continuationVisited;
    const profiles = payload.continuationProfiles?.length
      ? payload.continuationProfiles
      : [{beamProfile: "detour", weight: 3.5, topologyWeight: 0.6}];
    const attempts = Math.min(payload.continuationAttempts || 8, probeCheckpoints.length);
    for (let index = 0; index < attempts && remainingVisited > 0; index++) {
      const checkpoint = probeCheckpoints[index];
      const remainingBound = (payload.upperBound || maxDepth) - checkpoint.cost;
      const attemptVisited = Math.ceil(remainingVisited / (attempts - index));
      const continuation = beamSearch({
        ...payload,
        ...profiles[index % profiles.length],
        preparedBoard: board,
        state: {
          rows: board.rows,
          robot: checkpoint.robot,
          boxes: checkpoint.boxes.map(([y, x, label]) => [pkey(y, x), label]),
        },
        upperBound: remainingBound,
        maxDepth: remainingBound,
        maxVisited: attemptVisited,
        beamWidth: payload.continuationWidth || 36,
        transpositionLimit: payload.continuationTranspositionLimit || 10000,
        seed: seed + (index + 1) * 32452843,
        progressOffset: (payload.progressOffset || 0) + visited,
        continuationVisited: 0,
        endgameVisited: 0,
      });
      if (continuation.path) {
        return {
          path: [...reconstructNodePath(checkpoint.node), ...continuation.path],
          visited: visited + continuation.visited,
          generated: generated + continuation.generated,
          retained: seenDepth.size + continuation.retained,
          peakFrontier: Math.max(peakFrontier, continuation.peakFrontier),
          transpositionEvictions:
            seenDepth.evictions +
            seenExactDepth.evictions +
            continuation.transpositionEvictions,
          bestEstimate: 0,
          bestPushes: checkpoint.cost,
          continuation: true,
        };
      }
      visited += continuation.visited;
      remainingVisited -= continuation.visited;
      if ((continuation.bestEstimate ?? Infinity) < bestEstimate) {
        bestEstimate = continuation.bestEstimate;
        bestPushes = checkpoint.cost + (continuation.bestPushes || 0);
      }
    }
  }
  if (payload.endgameVisited && probeCheckpoints.length) {
    let remainingVisited = payload.endgameVisited;
    const attempts = Math.min(payload.endgameAttempts || 12, probeCheckpoints.length);
    for (let index = 0; index < attempts && remainingVisited > 0; index++) {
      const checkpoint = probeCheckpoints[index];
      const remainingBound = (payload.upperBound || maxDepth) - checkpoint.cost;
      const attemptVisited = Math.ceil(remainingVisited / (attempts - index));
    const endgame = boundedPushDepthFirstSearch({
      algorithm: "bounded-push-dfs",
      preparedBoard: board,
      state: {
        rows: board.rows,
        robot: checkpoint.robot,
        boxes: checkpoint.boxes.map(([y, x, label]) => [pkey(y, x), label]),
      },
      upperBound: remainingBound,
      maxDepth: remainingBound,
      maxVisited: attemptVisited,
      transpositionLimit: payload.endgameTranspositionLimit || 30000,
      dfsProfile: payload.endgameProfiles?.[index % payload.endgameProfiles.length] ||
        payload.endgameProfile || "balanced",
      diversity: payload.diversity,
      seed: seed + 15485863,
      progressOffset: (payload.progressOffset || 0) + visited,
      forcedMacros: false,
    });
    if (endgame.path) {
      return {
        path: [...reconstructNodePath(checkpoint.node), ...endgame.path],
        visited: visited + endgame.visited,
        generated: generated + (endgame.generated || 0),
        retained: seenDepth.size + (endgame.retained || 0),
        peakFrontier: Math.max(peakFrontier, endgame.peakFrontier || 0),
        transpositionEvictions:
          seenDepth.evictions +
          seenExactDepth.evictions +
          (endgame.transpositionEvictions || 0),
        bestEstimate: 0,
        bestPushes: checkpoint.cost,
        endgame: true,
      };
    }
    visited += endgame.visited;
      remainingVisited -= endgame.visited;
    }
  }
  return {
    path: null,
    visited,
    generated,
    retained: seenDepth.size,
    peakFrontier,
    transpositionEvictions: seenDepth.evictions + seenExactDepth.evictions,
    cutoff: beamCutoff,
    terminationReason: beamCutoff ? "budget" : "frontier-exhausted",
    bestEstimate,
    bestPushes,
    trackedThrough,
    checkpoint: serializeSearchCheckpoint(bestHandoff, board),
    checkpoints: [...handoffCheckpoints.values()]
      .sort((left, right) =>
        left.estimate - right.estimate ||
        (left.cost + left.estimate) - (right.cost + right.estimate))
      .slice(0, handoffLimit)
      .map(checkpoint => serializeSearchCheckpoint(checkpoint, board)),
    phaseCheckpoint: serializeSearchCheckpoint(phaseHandoff, board),
  };
}

function beamRestartSearch(payload) {
  const restartCount = payload.restartCount || 3;
  const restartVisited = payload.restartVisited || 180000;
  const seedStride = payload.seedStride || 104729;
  const profiles = payload.restartProfiles?.length ? payload.restartProfiles : [{}];
  const preparedBoard = parse(payload.state);
  let visited = 0, bestEstimate = Infinity, bestPushes = 0;
  for (let restart = 0; restart < restartCount; restart++) {
    const result = beamSearch({
      ...payload,
      ...profiles[restart % profiles.length],
      algorithm: "push-beam",
      preparedBoard,
      maxVisited: restartVisited,
      progressOffset: visited,
      seed: (payload.seed || 0) + restart * seedStride,
    });
    visited += result.visited;
    if ((result.bestEstimate ?? Infinity) < bestEstimate) {
      bestEstimate = result.bestEstimate;
      bestPushes = result.bestPushes || 0;
    }
    if (result.path) return {...result, visited, restart: restart + 1};
  }
  return {path: null, visited, cutoff: true, terminationReason: "restart-budget",
    bestEstimate, bestPushes, restarts: restartCount};
}

function boundedPushDepthFirstSearch(payload) {
  const board = payload.preparedBoard || parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const bound = payload.upperBound ?? payload.pushBound ?? 300;
  const maxVisited = payload.maxVisited || 250000;
  const maxDepth = payload.maxDepth || bound;
  const seed = payload.seed || 0;
  const profile = payload.dfsProfile || "balanced";
  const discrepancyLimit = payload.discrepancyLimit ?? Infinity;
  const lockProvenCommitments = payload.lockProvenCommitments !== false;
  const transpositions = new BoundedDepthMap(payload.transpositionLimit || 60000);
  const activePath = new Set(), segments = [];
  const checkpointLimit = payload.checkpointLimit || 8;
  const checkpoints = new Map();
  let visited = 0, reported = 0, cutoff = false, solution = null;
  const progressInterval = payload.progressInterval || 5000;
  const progressIntervalMs = payload.progressIntervalMs || 5000;
  let lastProgressAt = now();
  let bestEstimate = Infinity, bestPushes = 0;
  let bestCheckpoint = null;
  let trackedThrough = payload.trackedSignatures ? 0 : undefined;

  const visit = (state, cost, discrepancyRemaining) => {
    if (cutoff || solution) return;
    visited++;
    if (visited >= maxVisited) {
      cutoff = true;
      return;
    }
    const progressNow = now();
    if (visited - reported >= progressInterval ||
        progressNow - lastProgressAt >= progressIntervalMs) {
      postMessage({type: "progress", visited: (payload.progressOffset || 0) + visited,
        bestEstimate, bestPushes, depth: cost, retained: transpositions.size,
        performance: performanceSnapshot(board.metrics)});
      reported = visited;
      lastProgressAt = progressNow;
    }
    if (goal(state.boxes, board.goals)) {
      solution = segments.flatMap(segment => segment);
      return;
    }
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) return;
    const identity = pushIdentity(state, reachable);
    if (payload.trackedSignatures &&
        payload.trackedSignatures[cost] === pushKey(state, reachable)) {
      trackedThrough = Math.max(trackedThrough, cost);
    }
    if (activePath.has(identity) || (transpositions.get(identity) ?? Infinity) <= cost) return;
    activePath.add(identity);
    transpositions.set(identity, cost);

    const dependencyGraph = supportDependencyGraph(state, board, reachable);
    const localRooms = [
      ...exactLocalRoomAnalyses(state, board, reachable),
      ...exactLocalCorralAnalyses(state, board, reachable),
    ];
    const doorwayBefore = typedDoorwayFlow(state.boxes, board);
    const goalAccessBefore = payload.goalAccessOrdering === false
      ? null : goalAccessAnalysis(state.boxes, board);
    const currentCommitments = lockProvenCommitments ? goalCommitments(state.boxes, board, {
      doorway: doorwayBefore,
      supportDependency: dependencyGraph,
      localAnalyses: localRooms,
    }) : null;
    const candidates = [];
    for (const rawNext of pushNeighbors(
      state,
      board,
      reachable,
      {commitments: currentCommitments},
    )) {
      const next = expandPushMacro(
        rawNext,
        board,
        payload.forcedMacros !== false,
        {lockProven: lockProvenCommitments},
      );
      if (!next) continue;
      const childCost = cost + next.pushes;
      if (childCost > maxDepth || childCost > bound) continue;
      const estimate = heuristic(next.boxes, board);
      if (!Number.isFinite(estimate) || childCost + estimate > bound) continue;
      const checkpointIdentity = exactPushIdentity(next, board);
      if (!checkpoints.has(checkpointIdentity)) {
        checkpoints.set(checkpointIdentity, {
          state: {
            rows: board.rows,
            robot: next.robot,
            boxes: next.boxes.map(([y, x, label]) => [pkey(y, x), label]),
          },
          path: [...segments.flatMap(segment => segment), ...next.path],
          cost: childCost,
          estimate,
        });
        if (checkpoints.size > checkpointLimit * 3) {
          const retained = [...checkpoints.entries()]
            .sort(([, left], [, right]) =>
              left.estimate - right.estimate ||
              (left.cost + left.estimate) - (right.cost + right.estimate))
            .slice(0, checkpointLimit);
          checkpoints.clear();
          retained.forEach(([retainedIdentity, checkpoint]) =>
            checkpoints.set(retainedIdentity, checkpoint));
        }
      }
      if (estimate < bestEstimate) {
        bestEstimate = estimate;
        bestPushes = childCost;
        bestCheckpoint = {
          state: {
            rows: board.rows,
            robot: next.robot,
            boxes: next.boxes.map(([y, x, label]) => [pkey(y, x), label]),
          },
          path: [...segments.flatMap(segment => segment), ...next.path],
          cost: childCost,
          estimate,
        };
      }
      const topology = topologyPenalty(next.boxes, board);
      const evacuation = profile === "evacuation" ? roomEvacuationPenalty(next.boxes, board) : 0;
      const dependencyDelta = supportDependencyDelta(dependencyGraph, next);
      const localRoomDelta = localRoomOrderingDelta(localRooms, next);
      const doorway = typedDoorwayFlow(next.boxes, board);
      const packing = goalPackingBonus(next.boxes, board, {
        doorway,
        supportDependency: dependencyGraph,
        localAnalyses: localRooms,
        transition: next,
      });
      const doorwayDelta = doorwayFlowDelta(doorwayBefore, state, next);
      const relevance = relevanceOrderingScore(state, board, next, {
        supportDependency: dependencyGraph,
        doorway: doorwayBefore,
        goalAccess: goalAccessBefore,
        recentPush: state.recentPush,
      });
      const relevanceScore = recordRelevanceOrdering(board.metrics, relevance);
      let score = 2.5 * estimate + topology - 0.8 * packing;
      if (profile === "detour") score = 1.5 * estimate + 1.4 * topology - packing;
      if (profile === "setup" && childCost <= 12) score = -estimate + topology - packing;
      if (profile === "room-flow") score = estimate + 6 * topology - packing;
      if (profile === "evacuation") score = estimate + 8 * evacuation + topology - packing;
      score += (payload.supportDependencyWeight ?? 0.8) * dependencyDelta;
      score += (payload.localRoomWeight ?? 0.6) * localRoomDelta;
      score += (payload.doorwayFlowWeight ?? 0.35) *
        (0.2 * doorway.penalty + doorwayDelta);
      score += (payload.relevanceWeight ?? 0.6) * relevanceScore;
      score += (payload.diversity ?? 1.5) *
        signatureNoise(exactPushIdentity(next, board), seed + childCost);
      candidates.push({next, cost: childCost, score, relevance: relevance.signals});
    }
    candidates.sort((left, right) => left.score - right.score);
    for (let index = 0; index < candidates.length; index++) {
      const candidate = candidates[index];
      const discrepancy = index === 0 ? 0 : Math.ceil(Math.log2(index + 1));
      if (discrepancy > discrepancyRemaining) continue;
      segments.push(candidate.next.path);
      visit(
        {
          robot: candidate.next.robot,
          boxes: candidate.next.boxes,
          recentPush: {
            pushedFrom: candidate.next.pushedFrom,
            pushedTo: candidate.next.pushedTo,
          },
        },
        candidate.cost,
        discrepancyRemaining - discrepancy,
      );
      segments.pop();
      if (cutoff || solution) break;
    }
    activePath.delete(identity);
  };

  const initialEstimate = heuristic(initial.boxes, board);
  bestEstimate = initialEstimate;
  if (Number.isFinite(initialEstimate) && initialEstimate <= bound) {
    visit(initial, 0, discrepancyLimit);
  }
  return {
    path: solution,
    visited,
    cutoff,
    terminationReason: solution ? "solution" : cutoff ? "budget" : "profile-exhausted",
    bestEstimate,
    bestPushes,
    bound,
    discrepancyLimit,
    retained: transpositions.size,
    trackedThrough,
    checkpoint: bestCheckpoint,
    checkpoints: [...checkpoints.values()]
      .sort((left, right) =>
        left.estimate - right.estimate ||
        (left.cost + left.estimate) - (right.cost + right.estimate))
      .slice(0, checkpointLimit),
  };
}

const EXACT_CHECKPOINT_VERSION = 1;

function exactProblemHash(state) {
  const boxes = [...state.boxes].map(([position, label]) => `${label}@${position}`).sort();
  const source = `${state.rows.join("\n")}|r:${state.robot.join(",")}|b:${boxes.join(";")}`;
  let hash = 2166136261;
  for (let index = 0; index < source.length; index++) {
    hash ^= source.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function encodeExactIdentity(identity) {
  if (identity === null || identity === undefined) return null;
  return typeof identity === "bigint" ? `b:${identity}` : `s:${identity}`;
}

function decodeExactIdentity(identity) {
  if (identity === null || identity === undefined) return null;
  return identity.startsWith("b:") ? BigInt(identity.slice(2)) : identity.slice(2);
}

function pushIterativeDeepeningAStar(payload) {
  const board = parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const upperBound = payload.upperBound ?? payload.pushBound ?? 300;
  const maxVisited = payload.maxVisited || 300000;
  const seed = payload.seed || 0;
  const profile = payload.idaProfile || "balanced";
  const exactShard = payload.exactShard;
  const problemHash = exactProblemHash(payload.state);
  const checkpointBuild = payload.solverBuild || "development";
  const lockProvenCommitments = payload.lockProvenCommitments !== false;
  const progressInterval = payload.progressInterval || 25000;
  const resume = payload.resumeExactCheckpoint;
  if (resume && (
    resume.version !== EXACT_CHECKPOINT_VERSION ||
    resume.problemHash !== problemHash ||
    resume.solverBuild !== checkpointBuild ||
    JSON.stringify(resume.exactShard || null) !== JSON.stringify(exactShard || null) ||
    resume.upperBound !== (Number.isFinite(upperBound) ? upperBound : "Infinity")
  )) {
    return {path: null, visited: 0, cutoff: false, failed: true,
      terminationReason: "checkpoint-incompatible", checkpointRejected: true};
  }

  let visited = resume?.visited || 0, reported = visited, solution = null, cutoff = false;
  let bestEstimate = resume?.bestEstimate ?? heuristic(initial.boxes, board);
  let bestPushes = resume?.bestPushes || 0;
  let generated = resume?.generated || 0, thresholdPrunes = resume?.thresholdPrunes || 0;
  let upperBoundPrunes = resume?.upperBoundPrunes || 0;
  let corralPrunes = resume?.corralPrunes || 0, cyclePrunes = resume?.cyclePrunes || 0;
  let transpositionPrunes = resume?.transpositionPrunes || 0;
  let shardRejections = resume?.shardRejected || 0, shardAcceptances = resume?.shardAccepted || 0;
  let maxDepth = resume?.maxDepth || 0;
  let transpositionEvictions = resume?.transpositionEvictions || 0;
  let maxTranspositions = resume?.maxTranspositions || 0;
  let nextThresholdCandidate = resume?.nextThreshold ?? Infinity;
  let trackedThrough = payload.trackedSignatures ? (resume?.trackedThrough || 0) : undefined;
  let threshold = resume?.threshold ?? bestEstimate;
  let stack = (resume?.stack || []).map(frame => ({
    ...frame,
    identity: decodeExactIdentity(frame.identity),
  }));
  let transpositions = new BoundedDepthMap(payload.transpositionLimit || 80000);
  if (resume?.transpositions) {
    for (const [identity, cost] of resume.transpositions) {
      transpositions.set(decodeExactIdentity(identity), cost);
    }
    transpositions.evictions = resume.currentTranspositionEvictions || 0;
  }
  const activePath = new Set(stack.filter(frame => frame.entered).map(frame => frame.identity));
  const pauseAt = payload.pauseAfterVisited
    ? visited + payload.pauseAfterVisited : Infinity;
  const strategicOrderingGate = createOrderingProductivityGate(
    payload.strategicOrderingWarmup || 64,
    payload.strategicOrderingCooldown || 512,
  );
  if (!Number.isFinite(bestEstimate)) {
    return {path: null, visited, cutoff: false, terminationReason: "infeasible-root",
      bestEstimate, bestPushes};
  }

  const orderingScore = (state, cost) => {
    const estimate = heuristic(state.boxes, board);
    const topology = topologyPenalty(state.boxes, board);
    if (profile === "milestone") {
      return estimate + 2.5 * topology +
        0.35 * signatureNoise(roomFlowSignature(state.boxes, board), seed + cost);
    }
    if (profile === "detour") return estimate + 1.25 * topology;
    return estimate + 0.6 * topology;
  };
  const makeRootFrame = () => ({
    state: initial, cost: 0, accepted: false, entered: false,
    identity: null, candidates: null, nextIndex: 0, pathFromParent: [],
  });
  const makeCheckpoint = () => ({
    version: EXACT_CHECKPOINT_VERSION,
    problemHash,
    solverBuild: checkpointBuild,
    exactShard: exactShard || null,
    upperBound: Number.isFinite(upperBound) ? upperBound : "Infinity",
    threshold,
    visited,
    bestEstimate,
    bestPushes,
    generated,
    thresholdPrunes,
    upperBoundPrunes,
    corralPrunes,
    cyclePrunes,
    transpositionPrunes,
    shardRejected: shardRejections,
    shardAccepted: shardAcceptances,
    maxDepth,
    transpositionEvictions,
    maxTranspositions,
    nextThreshold: Number.isFinite(nextThresholdCandidate) ? nextThresholdCandidate : null,
    trackedThrough,
    stack: stack.map(frame => ({
      ...frame,
      identity: encodeExactIdentity(frame.identity),
      state: {
        robot: [...frame.state.robot],
        boxes: frame.state.boxes.map(box => [...box]),
        recentPush: frame.state.recentPush ? {...frame.state.recentPush} : undefined,
      },
      candidates: frame.candidates?.map(candidate => ({
        ...candidate,
        state: {
          robot: [...candidate.state.robot],
          boxes: candidate.state.boxes.map(box => [...box]),
          recentPush: candidate.state.recentPush
            ? {...candidate.state.recentPush} : undefined,
        },
        path: [...candidate.path],
      })) || null,
      pathFromParent: [...frame.pathFromParent],
    })),
    // Transpositions are a performance cache, not proof progress. Retaining only
    // a recent bounded tail keeps durable checkpoints within browser storage
    // limits; omitted entries can only cause repeated work after resume.
    transpositions: [...transpositions.values].slice(-2000)
      .map(([identity, cost]) => [encodeExactIdentity(identity), cost]),
    currentTranspositionEvictions: transpositions.evictions,
  });
  const emitProgress = (cost, includeCheckpoint = false) => {
    postMessage({type: "progress", visited, threshold, bestEstimate, bestPushes,
      depth: cost, maxDepth, generated, thresholdPrunes, upperBoundPrunes,
      corralPrunes, cyclePrunes, transpositionPrunes,
      shardRejected: shardRejections, shardAccepted: shardAcceptances,
      transpositions: transpositions.size, transpositionEvictions: transpositions.evictions,
      nextThreshold: Number.isFinite(nextThresholdCandidate) ? nextThresholdCandidate : undefined,
      exactCheckpoint: includeCheckpoint ? makeCheckpoint() : undefined,
      performance: performanceSnapshot(board.metrics)});
    reported = visited;
  };

  while (Number.isFinite(threshold) && threshold <= upperBound && !solution && !cutoff) {
    if (!stack.length) {
      postMessage({type: "contour", threshold, visited, exactShard});
      activePath.clear();
      transpositions = new BoundedDepthMap(payload.transpositionLimit || 80000);
      nextThresholdCandidate = Infinity;
      stack = [makeRootFrame()];
    }
    while (stack.length && !solution && !cutoff) {
      const frame = stack[stack.length - 1];
      const {state, cost} = frame;
      maxDepth = Math.max(maxDepth, cost);
      if (!frame.entered) {
        const estimate = heuristic(state.boxes, board);
        const total = cost + estimate;
        if (!frame.accepted && exactShard && cost >= exactShard.depth) {
          const bucket = Math.floor(
            signatureNoise(exactPushIdentity(state, board), 0) * exactShard.count,
          );
          if (bucket !== exactShard.index) {
            shardRejections++;
            stack.pop();
            continue;
          }
          frame.accepted = true;
          shardAcceptances++;
        }
        if (total > threshold) {
          thresholdPrunes++;
          nextThresholdCandidate = Math.min(nextThresholdCandidate, total);
          stack.pop();
          continue;
        }
        if (goal(state.boxes, board.goals)) {
          solution = stack.flatMap(candidate => candidate.pathFromParent);
          break;
        }
        visited++;
        if (visited >= maxVisited) {
          cutoff = true;
          break;
        }
        const reachable = reachablePaths(state, board);
        if (createsSealedCorralDeadlock(state, board, reachable)) {
          corralPrunes++;
          stack.pop();
          continue;
        }
        const identity = pushIdentity(state, reachable);
        if (payload.trackedSignatures &&
            payload.trackedSignatures[cost] === pushKey(state, reachable)) {
          trackedThrough = Math.max(trackedThrough, cost);
        }
        if (activePath.has(identity)) {
          cyclePrunes++;
          stack.pop();
          continue;
        }
        if ((transpositions.get(identity) ?? Infinity) <= cost) {
          transpositionPrunes++;
          stack.pop();
          continue;
        }
        activePath.add(identity);
        transpositions.set(identity, cost);
        frame.identity = identity;
        frame.entered = true;
        const dependencyGraph = supportDependencyGraph(state, board, reachable);
        const localRooms = [
          ...exactLocalRoomAnalyses(state, board, reachable),
          ...exactLocalCorralAnalyses(state, board, reachable),
        ];
        const doorwayBefore = typedDoorwayFlow(state.boxes, board);
        const goalAccessBefore = payload.goalAccessOrdering === false
          ? null : goalAccessAnalysis(state.boxes, board);
        const commitments = lockProvenCommitments ? goalCommitments(state.boxes, board, {
          doorway: doorwayBefore,
          supportDependency: dependencyGraph,
          localAnalyses: localRooms,
        }) : null;
        const candidates = [];
        for (const rawNext of pushNeighbors(state, board, reachable, {commitments})) {
          generated++;
          const next = expandPushMacro(rawNext, board, payload.forcedMacros !== false,
            {lockProven: lockProvenCommitments});
          if (!next) continue;
          const childCost = cost + next.pushes;
          const childEstimate = heuristic(next.boxes, board);
          if (childCost > upperBound || !Number.isFinite(childEstimate) ||
              childCost + childEstimate > upperBound) {
            upperBoundPrunes++;
            continue;
          }
          if (childEstimate < bestEstimate) {
            bestEstimate = childEstimate;
            bestPushes = childCost;
          }
          const doorwayDelta = doorwayFlowDelta(doorwayBefore, state, next);
          const relevance = relevanceOrderingScore(state, board, next, {
            supportDependency: dependencyGraph,
            doorway: doorwayBefore,
            goalAccess: goalAccessBefore,
            recentPush: state.recentPush,
          });
          const relevanceScore = recordRelevanceOrdering(board.metrics, relevance);
          const baseScore = orderingScore(next, childCost) +
            (payload.supportDependencyWeight ?? 0.5) *
              supportDependencyDelta(dependencyGraph, next) +
            (payload.localRoomWeight ?? 0.4) * localRoomOrderingDelta(localRooms, next) +
            (payload.doorwayFlowWeight ?? 0.25) * doorwayDelta +
            (payload.relevanceWeight ?? 0.6) * relevanceScore +
            (payload.diversity ?? 0.2) *
              signatureNoise(exactPushIdentity(next, board), seed + childCost);
          candidates.push({
            state: {
              robot: next.robot,
              boxes: next.boxes,
              recentPush: {pushedFrom: next.pushedFrom, pushedTo: next.pushedTo},
            },
            path: next.path,
            cost: childCost,
            total: childCost + childEstimate,
            baseScore,
            score: baseScore,
            relevance: relevance.signals,
          });
        }
        const orderable = candidates.filter(candidate => candidate.total <= threshold);
        if (orderable.length > 1 && strategicOrderingGate.shouldEvaluate()) {
          board.metrics.strategicOrderingEvaluations++;
          const baseline = [...orderable].sort((left, right) =>
            left.total - right.total || left.baseScore - right.baseScore);
          const packingWeight = ["milestone", "detour"].includes(profile) ? 1 : 0.8;
          const doorwayWeight = payload.doorwayFlowWeight ?? 0.25;
          for (const candidate of orderable) {
            const candidateState = {...candidate.state, path: candidate.path};
            const doorway = typedDoorwayFlow(candidate.state.boxes, board);
            const packing = goalPackingBonus(candidate.state.boxes, board, {
              doorway,
              supportDependency: dependencyGraph,
              localAnalyses: localRooms,
              transition: candidateState,
            });
            candidate.score += 0.2 * doorwayWeight * doorway.penalty - packingWeight * packing;
          }
          const enriched = [...orderable].sort((left, right) =>
            left.total - right.total || left.score - right.score);
          const changed = baseline.some((candidate, index) => candidate !== enriched[index]);
          if (changed) {
            board.metrics.strategicOrderingChanges++;
            const changedIndex = enriched.findIndex(
              (candidate, index) => candidate !== baseline[index]);
            enriched[changedIndex].orderingProbe = {baselineBest: bestEstimate};
          } else {
            strategicOrderingGate.observe({changed: false, useful: false});
          }
        } else if (orderable.length > 1) {
          board.metrics.strategicOrderingSkips++;
        }
        candidates.sort((left, right) => left.total - right.total || left.score - right.score);
        frame.candidates = candidates;
        if (visited >= pauseAt) {
          cutoff = true;
          break;
        }
      }
      if (visited - reported >= progressInterval) emitProgress(cost, true);
      let descended = false;
      while (frame.nextIndex < frame.candidates.length) {
        const candidate = frame.candidates[frame.nextIndex++];
        if (candidate.total > threshold) {
          thresholdPrunes++;
          nextThresholdCandidate = Math.min(nextThresholdCandidate, candidate.total);
          continue;
        }
        stack.push({
          state: candidate.state,
          cost: candidate.cost,
          accepted: frame.accepted,
          entered: false,
          identity: null,
          candidates: null,
          nextIndex: 0,
          pathFromParent: candidate.path,
          orderingProbe: candidate.orderingProbe || null,
        });
        descended = true;
        break;
      }
      if (descended) continue;
      if (frame.orderingProbe) {
        const useful = bestEstimate < frame.orderingProbe.baselineBest;
        strategicOrderingGate.observe({changed: true, useful});
        if (useful) board.metrics.strategicOrderingUseful++;
        if (strategicOrderingGate.snapshot().cooldownRemaining) {
          board.metrics.strategicOrderingCooldowns++;
        }
      }
      activePath.delete(frame.identity);
      stack.pop();
    }
    if (cutoff || solution) break;
    transpositionEvictions += transpositions.evictions;
    maxTranspositions = Math.max(maxTranspositions, transpositions.size);
    stack = [];
    if (!Number.isFinite(nextThresholdCandidate)) break;
    threshold = nextThresholdCandidate <= threshold ? threshold + 1 : nextThresholdCandidate;
  }
  const paused = cutoff && visited >= pauseAt && visited < maxVisited;
  return {
    path: solution,
    visited,
    cutoff,
    terminationReason: solution ? "solution" : paused ? "checkpoint-yield" :
      cutoff ? "budget" : "bound-exhausted",
    exactCheckpoint: paused ? makeCheckpoint() : undefined,
    bestEstimate,
    bestPushes,
    threshold,
    bound: upperBound,
    trackedThrough,
    exactShard,
    generated,
    thresholdPrunes,
    upperBoundPrunes,
    corralPrunes,
    cyclePrunes,
    transpositionPrunes,
    shardRejected: shardRejections,
    shardAccepted: shardAcceptances,
    transpositionEvictions: transpositionEvictions + transpositions.evictions,
    maxTranspositions: Math.max(maxTranspositions, transpositions.size),
    maxDepth,
    nextThreshold: Number.isFinite(nextThresholdCandidate) ? nextThresholdCandidate : undefined,
  };
}

function bridgeAStarSearch(payload) {
  const board = parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
    cost: 0,
  };
  const targetBoxes = payload.targetState.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  const targetState = {robot: payload.targetState.robot, boxes: targetBoxes};
  const targetReachable = reachablePaths(targetState, board);
  const targetKey = payload.targetId || pushKey(targetState, targetReachable);
  const heuristicMemo = new Map();
  const frontier = new Heap(), bestCost = new Map(), closed = new Set();
  let cameFrom = new Map();
  const weight = payload.weight || 1.4;
  const maxVisited = payload.maxVisited || 100000;
  const frontierLimit = payload.frontierLimit || 4000;
  let visited = 0, order = 0, bestEstimate = Infinity, bestCheckpoint = null;
  let compactions = 0, peakFrontier = 0;

  initial.reachable = reachablePaths(initial, board);
  initial.signature = pushKey(initial, initial.reachable);
  initial.estimate = targetLayoutHeuristic(initial.boxes, targetBoxes, board, heuristicMemo);
  const initialEstimate = initial.estimate;
  if (!Number.isFinite(initial.estimate)) {
    return {path: null, visited, cutoff: false,
      terminationReason: "target-incompatible", bestEstimate: initial.estimate};
  }
  delete initial.reachable;
  bestCost.set(initial.signature, 0);
  frontier.push([weight * initial.estimate, order++, initial]);

  while (frontier.length) {
    const current = frontier.pop()[2];
    if (bestCost.get(current.signature) !== current.cost || closed.has(current.signature)) continue;
    bestCost.delete(current.signature);
    closed.add(current.signature);
    visited++;
    if (current.signature === targetKey) {
      return {
        path: reconstructPath(cameFrom, current.signature),
        visited,
        terminationReason: "target-reached",
        initialEstimate,
        bestEstimate: 0,
        bestPushes: current.cost,
        peakFrontier,
        compactions,
        finalState: {
          rows: board.rows,
          robot: current.robot,
          boxes: current.boxes.map(([y, x, label]) => [pkey(y, x), label]),
        },
      };
    }
    if (visited >= maxVisited) break;
    const currentReachable = reachablePaths(current, board);
    for (const rawNext of pushNeighbors(current, board, currentReachable)) {
      const next = expandPushMacro(rawNext, board, payload.forcedMacros !== false);
      if (!next) continue;
      const child = {
        robot: next.robot,
        boxes: next.boxes,
        cost: current.cost + next.pushes,
      };
      if (payload.upperBound && child.cost > payload.upperBound) continue;
      child.reachable = reachablePaths(child, board);
      child.signature = pushKey(child, child.reachable);
      delete child.reachable;
      if (closed.has(child.signature) ||
          child.cost >= (bestCost.get(child.signature) ?? Infinity)) continue;
      child.estimate = targetLayoutHeuristic(child.boxes, targetBoxes, board, heuristicMemo);
      if (!Number.isFinite(child.estimate) ||
          (payload.upperBound && child.cost + child.estimate > payload.upperBound)) continue;
      if (child.estimate < bestEstimate) {
        bestEstimate = child.estimate;
        bestCheckpoint = {
          state: {
            rows: board.rows,
            robot: child.robot,
            boxes: child.boxes.map(([y, x, label]) => [pkey(y, x), label]),
          },
          cost: child.cost,
          estimate: child.estimate,
          signature: child.signature,
        };
      }
      bestCost.set(child.signature, child.cost);
      cameFrom.set(child.signature, {parent: current.signature, segment: next.path});
      frontier.push([child.cost + weight * child.estimate, order++, child]);
    }
    peakFrontier = Math.max(peakFrontier, frontier.length);
    if (frontier.length > frontierLimit * 2) {
      frontier.retainBest(frontierLimit);
      const retainedCosts = new Map();
      const ancestry = new Set([initial.signature]);
      const pending = [];
      for (const [, , state] of frontier.items) {
        const previous = retainedCosts.get(state.signature) ?? Infinity;
        if (state.cost < previous) retainedCosts.set(state.signature, state.cost);
        pending.push(state.signature);
      }
      if (bestCheckpoint?.signature) pending.push(bestCheckpoint.signature);
      while (pending.length) {
        const signature = pending.pop();
        if (ancestry.has(signature)) continue;
        ancestry.add(signature);
        const record = cameFrom.get(signature);
        if (record?.parent) pending.push(record.parent);
      }
      cameFrom = new Map([...cameFrom].filter(([signature]) => ancestry.has(signature)));
      bestCost.clear();
      retainedCosts.forEach((cost, signature) => bestCost.set(signature, cost));
      compactions++;
    }
    if (visited % 5000 === 0) postMessage({type: "progress", visited,
      bestEstimate, bestPushes: bestCheckpoint?.cost, frontier: frontier.length,
      retained: bestCost.size, peakFrontier, compactions,
      performance: performanceSnapshot(board.metrics)});
  }
  const cutoff = visited >= maxVisited;
  const checkpoint = bestCheckpoint && {
    state: bestCheckpoint.state,
    path: reconstructPath(cameFrom, bestCheckpoint.signature),
    cost: bestCheckpoint.cost,
    estimate: bestCheckpoint.estimate,
  };
  return {path: null, visited, cutoff,
    terminationReason: cutoff ? "budget" : "frontier-exhausted",
    initialEstimate, bestEstimate, bestPushes: bestCheckpoint?.cost,
    frontier: frontier.length, retained: bestCost.size, peakFrontier, compactions,
    checkpoint};
}

function bidirectionalSide(payload) {
  validatePuzzleRows(payload.state.rows);
  const board = parse(payload.state);
  const initialBoxes = payload.state.boxes.map(([p, label]) => [...p.split(",").map(Number), label]);
  const initialTargets = targetMapFromBoxes(initialBoxes, board);
  const forward = payload.mode === "bidir-forward";
  const frontier = new Heap(), closed = new Set(), records = [];
  let bestCost = new Map();
  const frontierLimit = payload.frontierLimit || 40000;
  let order = 0, visited = 0, reported = 0, bestLandmarkEstimate = Infinity;
  let generated = 0, peakFrontier = 0, compactions = 0;
  const compactFrontier = () => {
    peakFrontier = Math.max(peakFrontier, frontier.length);
    if (frontier.length <= frontierLimit * 2) return;
    frontier.retainBest(frontierLimit);
    const retainedCosts = new Map();
    for (const [, , state] of frontier.items) {
      const previous = retainedCosts.get(state.exactIdentity) ?? Infinity;
      if (state.cost < previous) retainedCosts.set(state.exactIdentity, state.cost);
    }
    bestCost = retainedCosts;
    compactions++;
  };
  const landmarkCandidates = new Map();
  const emitLandmarks = () => {
    if (forward || !landmarkCandidates.size) return;
    const landmarks = stratifiedCheckpoints([...landmarkCandidates.values()])
      .slice(0, payload.landmarkLimit || 64)
      .map(({checkpointBand: _band, checkpointClass: _class, ...landmark}) => landmark);
    postMessage({type: "landmarks", landmarks});
  };
  const starts = forward
    ? [{robot: payload.state.robot, boxes: initialBoxes, cost: 0, path: []}]
    : reverseStartStates(
      board,
      initialBoxes,
      payload.reverseShard || {index: 0, count: 1},
      initialTargets,
    );
  if (!forward) postMessage({
    type: "reverse-starts",
    shard: payload.reverseShard || {index: 0, count: 1},
    ...starts.portfolioStats,
  });
  starts.forEach(state => {
    state.exactIdentity = exactPushIdentity(state, board);
    const estimate = forward
      ? heuristic(state.boxes, board)
      : homeHeuristic(state.boxes, initialTargets);
    if (!Number.isFinite(estimate) || bestCost.has(state.exactIdentity)) return;
    if (payload.upperBound && state.cost + estimate > payload.upperBound) return;
    bestCost.set(state.exactIdentity, state.cost);
    const topology = forward ? 0.2 * topologyPenalty(state.boxes, board) : 0;
    frontier.push([state.cost + estimate + topology, order++, state]);
  });
  compactFrontier();

  while (frontier.length) {
    const current = frontier.pop()[2];
    if (bestCost.get(current.exactIdentity) !== current.cost) continue;
    bestCost.delete(current.exactIdentity);
    const reachable = reachablePaths(current, board);
    if (forward && createsSealedCorralDeadlock(current, board, reachable)) continue;
    const identity = pushIdentity(current, reachable);
    if (closed.has(identity)) continue;
    closed.add(identity); visited++;
    const signature = pushKey(current, reachable);
    records.push({
      id: signature,
      parent: current.parent ?? null,
      segment: encodeMoves(current.segment || []),
      robot: current.robot,
    });
    const landmarkEstimate = forward
      ? heuristic(current.boxes, board)
      : homeHeuristic(current.boxes, initialTargets);
    if (landmarkEstimate < bestLandmarkEstimate) {
      bestLandmarkEstimate = landmarkEstimate;
      postMessage({
        type: "landmark",
        id: signature,
        estimate: landmarkEstimate,
        cost: current.cost,
        state: {
          rows: board.rows,
          robot: current.robot,
          boxes: current.boxes.map(([y, x, label]) => [pkey(y, x), label]),
        },
      });
    }
    if (!forward) {
      const solvedGoals = current.boxes
        .filter(([y, x, label]) => board.goals.get(pkey(y, x)) === label)
        .map(([y, x, label]) => `${y},${x},${label}`)
        .sort()
        .join(";");
      const checkpointBand = Math.floor(landmarkEstimate / 10);
      const checkpointClass =
        `${checkpointBand}|${roomFlowSignature(current.boxes, board)}|${solvedGoals}`;
      const existing = landmarkCandidates.get(checkpointClass);
      if (!existing || landmarkEstimate < existing.estimate ||
          (landmarkEstimate === existing.estimate && current.cost < existing.cost)) {
        landmarkCandidates.set(checkpointClass, {
          id: signature,
          estimate: landmarkEstimate,
          cost: current.cost,
          checkpointBand,
          checkpointClass,
          state: {
            rows: board.rows,
            robot: current.robot,
            boxes: current.boxes.map(([y, x, label]) => [pkey(y, x), label]),
          },
        });
      }
    }

    if (records.length >= 500) flushRecords(records);
    if (payload.maxVisited && visited >= payload.maxVisited) {
      flushRecords(records);
      emitLandmarks();
      postMessage({type: "progress", visited, delta: visited - reported,
        bestEstimate: bestLandmarkEstimate, frontier: frontier.length,
        retained: bestCost.size, generated, peakFrontier, compactions,
        performance: performanceSnapshot(board.metrics)});
      postMessage({type: "done", visited, cutoff: true, terminationReason: "budget",
        bestEstimate: bestLandmarkEstimate, generated, peakFrontier, compactions,
        frontier: frontier.length, retained: bestCost.size,
        performance: performanceSnapshot(board.metrics)});
      return;
    }
    let nextStates = forward
      ? pushNeighbors(current, board, reachable).map(next => ({
          robot: next.robot,
          boxes: next.boxes,
          cost: current.cost + 1,
          parent: signature,
          segment: next.path,
        }))
      : reversePullNeighbors(current, board, reachable).map(next => ({
          ...next,
          parent: signature,
        }));
    if (!forward && current.cost === 0 && payload.reverseShard?.count > 1) {
      nextStates = nextStates.filter(next =>
        reverseShardOwns(exactPushIdentity(next, board), payload.reverseShard));
    }
    for (const next of nextStates) {
      next.exactIdentity = exactPushIdentity(next, board);
      if (next.cost >= (bestCost.get(next.exactIdentity) ?? Infinity)) continue;
      const estimate = forward
        ? heuristic(next.boxes, board)
        : homeHeuristic(next.boxes, initialTargets);
      if (!Number.isFinite(estimate)) continue;
      if (payload.upperBound && next.cost + estimate > payload.upperBound) continue;
      bestCost.set(next.exactIdentity, next.cost);
      generated++;
      const weightedEstimate = (forward ? 1.4 : 1.2) * estimate;
      const topology = forward ? 0.2 * topologyPenalty(next.boxes, board) : 0;
      frontier.push([next.cost + weightedEstimate + topology, order++, next]);
    }
    compactFrontier();
    if (visited % 1000 === 0) {
      postMessage({type: "progress", visited, delta: visited - reported,
        bestEstimate: bestLandmarkEstimate, frontier: frontier.length,
        retained: bestCost.size, generated, peakFrontier, compactions,
        performance: performanceSnapshot(board.metrics)});
      reported = visited;
    }
  }
  flushRecords(records);
  emitLandmarks();
  postMessage({type: "progress", visited, delta: visited - reported,
    bestEstimate: bestLandmarkEstimate, frontier: frontier.length,
    retained: bestCost.size, generated, peakFrontier, compactions,
    performance: performanceSnapshot(board.metrics)});
  postMessage({type: "done", visited, cutoff: false, terminationReason: "exhausted",
    bestEstimate: bestLandmarkEstimate, generated, peakFrontier, compactions,
    frontier: frontier.length, retained: bestCost.size,
    performance: performanceSnapshot(board.metrics)});
}

function goalCutComponentSolved(boxes, board, domain) {
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  return [...domain].every(position => {
    const goalLabel = board.goals.get(position);
    const boxLabel = occupied.get(position);
    return goalLabel ? boxLabel === goalLabel : !boxLabel;
  });
}

function replaySearchPath(state, board, path) {
  let replay = state;
  for (const move of path) {
    const next = neighbors(replay, board, false).find(candidate => candidate.move === move);
    if (!next) return null;
    replay = {robot: next.robot, boxes: next.boxes, cost: replay.cost + 1};
  }
  return replay;
}

function solveGoalCutComponents(payload, board, initial, certificate) {
  let current = initial, visited = 0;
  const path = [];
  const robotPosition = pkey(initial.robot[0], initial.robot[1]);
  const ordered = [...certificate.components].sort((left, right) =>
    Number(right.has(robotPosition)) - Number(left.has(robotPosition)));
  for (const domain of ordered) {
    if (goalCutComponentSolved(current.boxes, board, domain)) continue;
    const remainingBudget = payload.maxVisited
      ? Math.max(1, payload.maxVisited - visited)
      : undefined;
    const result = searchCore({
      ...payload,
      state: {
        rows: board.rows,
        robot: current.robot,
        boxes: current.boxes.map(([y, x, label]) => [pkey(y, x), label]),
      },
      maxVisited: remainingBudget,
      _goalCutDomain: domain,
      _skipGoalCut: true,
    });
    visited += result.visited || 0;
    if (!result.path) return null;
    current = replaySearchPath(current, board, result.path);
    if (!current) return null;
    path.push(...result.path);
  }
  return goal(current.boxes, board.goals)
    ? {path, visited, decompositionComponents: ordered.length,
      decompositionCut: certificate.cut}
    : null;
}

function searchCore(payload) {
  if (payload.algorithm === "analyze-puzzle") {
    return {path: null, visited: 0, analysis: analyzePuzzleForSearch(payload.state)};
  }
  if (payload.algorithm === "bridge-astar") return bridgeAStarSearch(payload);
  if (payload.algorithm === "plan-macro-beam") return canonicalPlanMacroBeamSearch(payload);
  if (payload.algorithm === "push-beam") return beamSearch(payload);
  if (payload.algorithm === "push-beam-restarts") return beamRestartSearch(payload);
  if (payload.algorithm === "bounded-push-dfs") return boundedPushDepthFirstSearch(payload);
  if (payload.algorithm === "push-ida-star") return pushIterativeDeepeningAStar(payload);
  if (["ultimate", "portfolio", "fast"].includes(payload.algorithm)) {
    const beam = beamSearch({...payload, algorithm: "push-beam"});
    if (beam.path) return {...beam, strategy: "Push Beam"};
    const greedy = search({...payload, algorithm: "push-greedy"});
    if (greedy.path) return {...greedy, strategy: "Push Greedy"};
    const weighted = search({...payload, algorithm: "weighted-push-astar"});
    if (weighted.path) return {...weighted, strategy: "Weighted Push A*"};
    return search({...payload, algorithm: "push-astar"});
  }
  const board = parse(payload.state), initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([p, label]) => [...p.split(",").map(Number), label]),
    cost: 0,
    parent: null,
    segment: [],
  };
  if (!payload._skipGoalCut && !payload.maxVisited &&
      payload.algorithm === "push-astar") {
    const certificate = goalCutDecomposition(initial.boxes, board);
    if (certificate) {
      const decomposed = solveGoalCutComponents(payload, board, initial, certificate);
      if (decomposed) return decomposed;
    }
  }
  const algorithm = payload.algorithm, frontier = new Heap(), seen = new Map(), cameFrom = new Map();
  const bestCost = new Map(), closed = new Set();
  const pushMacro = ["push-astar", "push-greedy", "weighted-push-astar"].includes(algorithm);
  const weight = algorithm === "weighted-push-astar" ? 1.6 : 1;
  let order = 0, visited = 0;
  const score = (s) => algorithm === "bfs" ? s.cost :
    algorithm === "dfs" ? -s.cost :
    ["greedy", "push-greedy"].includes(algorithm)
      ? heuristic(s.boxes, board) + 0.3 * topologyPenalty(s.boxes, board) :
    s.cost + weight * heuristic(s.boxes, board) +
      (algorithm === "weighted-push-astar" ? 0.15 * topologyPenalty(s.boxes, board) : 0);
  if (pushMacro) {
    initial.exactIdentity = exactPushIdentity(initial, board);
    bestCost.set(initial.exactIdentity, 0);
  }
  const initialScore = score(initial);
  if (!Number.isFinite(initialScore)) return {path: null, visited: 0};
  frontier.push([initialScore, order++, initial]);
  while (frontier.length) {
    const current = frontier.pop()[2];
    if (pushMacro && bestCost.get(current.exactIdentity) !== current.cost) continue;
    const reachable = pushMacro ? reachablePaths(current, board) : null;
    const identity = pushMacro ? pushIdentity(current, reachable) : exactPushIdentity(current, board);
    if (pushMacro) {
      if (closed.has(identity)) continue;
      closed.add(identity);
    } else {
      if (seen.has(identity) && seen.get(identity) <= current.cost) continue;
      seen.set(identity, current.cost);
    }
    visited++;
    if (current.parent !== null) {
      cameFrom.set(identity, {parent: current.parent, segment: current.segment});
    }
    if (payload._goalCutDomain
      ? goalCutComponentSolved(current.boxes, board, payload._goalCutDomain)
      : goal(current.boxes, board.goals)) {
      return {path: reconstructPath(cameFrom, identity), visited};
    }
    if (pushMacro && createsSealedCorralDeadlock(current, board, reachable)) continue;
    if (payload.maxVisited && visited >= payload.maxVisited) {
      return {path: null, visited, cutoff: true};
    }
    let nextStates = pushMacro ? pushNeighbors(current, board, reachable)
      .map(next => expandPushMacro(next, board, payload.forcedMacros !== false))
      .filter(Boolean) :
      neighbors(current, board).map(n => ({robot: n.robot, boxes: n.boxes, path: [n.move]}));
    if (payload._goalCutDomain && pushMacro) {
      nextStates = nextStates.filter(next =>
        payload._goalCutDomain.has(next.pushedFrom) &&
        payload._goalCutDomain.has(next.pushedTo));
    }
    for (const next of nextStates) {
      const child = {robot: next.robot, boxes: next.boxes,
        cost: current.cost + (pushMacro ? next.pushes : next.path.length),
        parent: identity, segment: next.path};
      if (pushMacro) {
        child.exactIdentity = exactPushIdentity(child, board);
        if (child.cost >= (bestCost.get(child.exactIdentity) ?? Infinity)) continue;
        const childScore = score(child);
        if (!Number.isFinite(childScore)) continue;
        if (payload.upperBound && child.cost + heuristic(child.boxes, board) > payload.upperBound) continue;
        bestCost.set(child.exactIdentity, child.cost);
        frontier.push([childScore, order++, child]);
      } else {
        frontier.push([score(child), order++, child]);
      }
    }
    if (visited % 10000 === 0) postMessage({type: "progress", visited,
      performance: performanceSnapshot(board.metrics)});
  }
  return {path: null, visited};
}

const TERMINAL_STATUS = Object.freeze({
  SOLVED: "solved",
  PROVEN_UNSOLVABLE: "proven-unsolvable",
  CUTOFF: "cutoff",
  CANCELLED: "cancelled",
  FAILED: "failed",
});

function validateSearchSolution(payload, candidatePath) {
  if (!Array.isArray(candidatePath)) {
    return {valid: false, reason: "missing-solution-path", path: null};
  }
  const board = parse(payload.state);
  let replay = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
    cost: 0,
  };
  const validated = [];
  if (goal(replay.boxes, board.goals)) return {valid: true, reason: "solution", path: []};
  for (const move of candidatePath) {
    const next = neighbors(replay, board, false).find(candidate => candidate.move === move);
    if (!next) return {valid: false, reason: "illegal-solution-path", path: null};
    replay = {robot: next.robot, boxes: next.boxes, cost: replay.cost + 1};
    validated.push(move);
    if (goal(replay.boxes, board.goals)) {
      return {valid: true, reason: "solution", path: validated};
    }
  }
  return {valid: false, reason: "incomplete-solution-path", path: null};
}

function terminalSearchResult(payload, result) {
  if (payload.algorithm === "bridge-astar" || payload.algorithm === "analyze-puzzle") {
    return result;
  }
  if (result.path !== null && result.path !== undefined) {
    const validation = validateSearchSolution(payload, result.path);
    if (!validation.valid) {
      return {...result, path: null, status: TERMINAL_STATUS.FAILED,
        terminationReason: validation.reason};
    }
    return {...result, path: validation.path, status: TERMINAL_STATUS.SOLVED,
      terminationReason: "solution"};
  }
  if (result.cancelled || result.terminationReason === "user-stop") {
    return {...result, status: TERMINAL_STATUS.CANCELLED,
      terminationReason: result.terminationReason || "user-stop"};
  }
  if (result.failed) {
    return {...result, status: TERMINAL_STATUS.FAILED,
      terminationReason: result.terminationReason || "search-failed"};
  }
  const exactAlgorithms = new Set(["push-ida-star", "push-astar", "astar", "bfs", "dfs"]);
  const effectiveBound = payload.algorithm === "push-ida-star"
    ? (payload.upperBound ?? payload.pushBound ?? 300)
    : (payload.upperBound ?? payload.pushBound);
  const finiteBound = Number.isFinite(effectiveBound);
  const proofComplete = exactAlgorithms.has(payload.algorithm) && !result.cutoff &&
    (!finiteBound || result.terminationReason === "infeasible-root");
  if (proofComplete) {
    return {...result, status: TERMINAL_STATUS.PROVEN_UNSOLVABLE,
      terminationReason: result.terminationReason || "frontier-exhausted"};
  }
  return {...result, status: TERMINAL_STATUS.CUTOFF,
    cutoff: true, terminationReason: result.terminationReason || "search-incomplete"};
}

function search(payload) {
  const parentPerformance = activePerformance;
  const metrics = parentPerformance || createPerformanceMetrics();
  const rootSearch = parentPerformance === null;
  const started = now();
  activePerformance = metrics;
  try {
    if (payload.state?.rows) validatePuzzleRows(payload.state.rows);
    const result = searchCore(payload);
    if (rootSearch) {
      metrics.totalMs = now() - started;
      metrics._startedAt = null;
    }
    const performance = performanceSnapshot(metrics);
    return terminalSearchResult(payload, {
      ...result,
      generated:
        result.generated ??
        Math.max(result.visited || 0, performance.pushCandidates || 0),
      retained: result.retained ?? result.visited ?? 0,
      peakFrontier:
        result.peakFrontier ?? result.frontier ?? (result.visited ? 1 : 0),
      transpositionEvictions: result.transpositionEvictions ?? 0,
      performance,
    });
  } finally {
    activePerformance = parentPerformance;
  }
}
