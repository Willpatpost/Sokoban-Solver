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
  let lastProgressAt = now();
  let beamCutoff = false;
  let bestCheckpoint = null;
  let bestHandoff = null;
  let phaseHandoff = null;
  const endgameCheckpoints = [];
  let trackedThrough = payload.trackedSignatures ? 0 : undefined;

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
        return {path: reconstructNodePath(current.node), visited};
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
          };
        }
        child.exactIdentity = exactPushIdentity(child, board);
        if ((seenExactDepth.get(child.exactIdentity) ?? Infinity) <= child.cost) continue;
        const estimate = heuristic(child.boxes, board);
        if (!Number.isFinite(estimate)) continue;
        if (payload.upperBound && child.cost + estimate > payload.upperBound) continue;
        if (estimate < bestEstimate) {
          bestEstimate = estimate;
          bestPushes = child.cost;
        }
        const topology = topologyPenalty(child.boxes, board);
        const dependencyDelta = supportDependencyDelta(dependencyGraph, next);
        const localRoomDelta = localRoomOrderingDelta(localRooms, next);
        const doorway = typedDoorwayFlow(child.boxes, board);
        const packing = goalPackingBonus(child.boxes, board, {
          doorway,
          supportDependency: dependencyGraph,
          localAnalyses: localRooms,
          transition: next,
        });
        const doorwayDelta = doorwayFlowDelta(doorwayBefore, current, next);
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
          doorwayFlowWeight * (0.2 * doorway.penalty + doorwayDelta) +
          diversity * signatureNoise(child.exactIdentity, seed);
        const exploreScore = topologyWeight * topology + evacuationWeight * evacuation -
          goalPackingWeight * packing +
          supportDependencyWeight * dependencyDelta +
          localRoomWeight * localRoomDelta +
          doorwayFlowWeight * (0.2 * doorway.penalty + doorwayDelta) +
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
      let score = 2.5 * estimate + topology - 0.8 * packing;
      if (profile === "detour") score = 1.5 * estimate + 1.4 * topology - packing;
      if (profile === "setup" && childCost <= 12) score = -estimate + topology - packing;
      if (profile === "room-flow") score = estimate + 6 * topology - packing;
      if (profile === "evacuation") score = estimate + 8 * evacuation + topology - packing;
      score += (payload.supportDependencyWeight ?? 0.8) * dependencyDelta;
      score += (payload.localRoomWeight ?? 0.6) * localRoomDelta;
      score += (payload.doorwayFlowWeight ?? 0.35) *
        (0.2 * doorway.penalty + doorwayDelta);
      score += (payload.diversity ?? 1.5) *
        signatureNoise(exactPushIdentity(next, board), seed + childCost);
      candidates.push({next, cost: childCost, score});
    }
    candidates.sort((left, right) => left.score - right.score);
    for (let index = 0; index < candidates.length; index++) {
      const candidate = candidates[index];
      const discrepancy = index === 0 ? 0 : Math.ceil(Math.log2(index + 1));
      if (discrepancy > discrepancyRemaining) continue;
      segments.push(candidate.next.path);
      visit(
        {robot: candidate.next.robot, boxes: candidate.next.boxes},
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
  const segments = [], activePath = new Set();
  let visited = 0, reported = 0, solution = null, cutoff = false;
  let bestEstimate = heuristic(initial.boxes, board), bestPushes = 0;
  let generated = 0, thresholdPrunes = 0, upperBoundPrunes = 0;
  let corralPrunes = 0, cyclePrunes = 0, transpositionPrunes = 0;
  let shardRejections = 0, shardAcceptances = 0, maxDepth = 0;
  let transpositionEvictions = 0, maxTranspositions = 0;
  let nextThresholdCandidate = Infinity;
  const progressInterval = payload.progressInterval || 25000;
  let trackedThrough = payload.trackedSignatures ? 0 : undefined;
  const exactShard = payload.exactShard;
  const lockProvenCommitments = payload.lockProvenCommitments !== false;
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

  const searchContour = (state, cost, threshold, transpositions, shardAccepted = false) => {
    if (cutoff || solution) return Infinity;
    maxDepth = Math.max(maxDepth, cost);
    const estimate = heuristic(state.boxes, board);
    const total = cost + estimate;
    let accepted = shardAccepted;
    if (!accepted && exactShard && cost >= exactShard.depth) {
      const bucket = Math.floor(signatureNoise(exactPushIdentity(state, board), 0) * exactShard.count);
      if (bucket !== exactShard.index) {
        shardRejections++;
        return Infinity;
      }
      accepted = true;
      shardAcceptances++;
    }
    if (total > threshold) {
      thresholdPrunes++;
      nextThresholdCandidate = Math.min(nextThresholdCandidate, total);
      return total;
    }
    if (goal(state.boxes, board.goals)) {
      solution = segments.flatMap(segment => segment);
      return total;
    }
    if (++visited >= maxVisited) {
      cutoff = true;
      return Infinity;
    }
    if (visited - reported >= progressInterval) {
      postMessage({type: "progress", visited, threshold, bestEstimate, bestPushes,
        depth: cost, maxDepth, generated, thresholdPrunes, upperBoundPrunes,
        corralPrunes, cyclePrunes, transpositionPrunes,
        shardRejected: shardRejections, shardAccepted: shardAcceptances,
        transpositions: transpositions.size, transpositionEvictions: transpositions.evictions,
        nextThreshold: Number.isFinite(nextThresholdCandidate) ? nextThresholdCandidate : undefined,
        performance: performanceSnapshot(board.metrics)});
      reported = visited;
    }
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) {
      corralPrunes++;
      return Infinity;
    }
    const identity = pushIdentity(state, reachable);
    if (payload.trackedSignatures &&
        payload.trackedSignatures[cost] === pushKey(state, reachable)) {
      trackedThrough = Math.max(trackedThrough, cost);
    }
    if (activePath.has(identity)) {
      cyclePrunes++;
      return Infinity;
    }
    if ((transpositions.get(identity) ?? Infinity) <= cost) {
      transpositionPrunes++;
      return Infinity;
    }
    activePath.add(identity);
    transpositions.set(identity, cost);

    const dependencyGraph = supportDependencyGraph(state, board, reachable);
    const localRooms = [
      ...exactLocalRoomAnalyses(state, board, reachable),
      ...exactLocalCorralAnalyses(state, board, reachable),
    ];
    const doorwayBefore = typedDoorwayFlow(state.boxes, board);
    const currentCommitments = lockProvenCommitments ? goalCommitments(state.boxes, board, {
      doorway: doorwayBefore,
      supportDependency: dependencyGraph,
      localAnalyses: localRooms,
    }) : null;
    let nextThreshold = Infinity;
    const candidates = [];
    for (const rawNext of pushNeighbors(
      state,
      board,
      reachable,
      {commitments: currentCommitments},
    )) {
      generated++;
      const next = expandPushMacro(
        rawNext,
        board,
        payload.forcedMacros !== false,
        {lockProven: lockProvenCommitments},
      );
      if (!next) continue;
      const childCost = cost + next.pushes;
      if (childCost > upperBound) {
        upperBoundPrunes++;
        continue;
      }
      const childEstimate = heuristic(next.boxes, board);
      if (!Number.isFinite(childEstimate) || childCost + childEstimate > upperBound) {
        upperBoundPrunes++;
        continue;
      }
      if (childEstimate < bestEstimate) {
        bestEstimate = childEstimate;
        bestPushes = childCost;
      }
      const doorwayDelta = doorwayFlowDelta(doorwayBefore, state, next);
      const baseScore = orderingScore(next, childCost) +
        (payload.supportDependencyWeight ?? 0.5) *
          supportDependencyDelta(dependencyGraph, next) +
        (payload.localRoomWeight ?? 0.4) * localRoomOrderingDelta(localRooms, next) +
        (payload.doorwayFlowWeight ?? 0.25) * doorwayDelta +
        (payload.diversity ?? 0.2) *
          signatureNoise(exactPushIdentity(next, board), seed + childCost);
      candidates.push({
        next,
        cost: childCost,
        total: childCost + childEstimate,
        baseScore,
        score: baseScore,
      });
    }
    const orderable = candidates.filter(candidate => candidate.total <= threshold);
    if (orderable.length > 1) {
      if (strategicOrderingGate.shouldEvaluate()) {
        board.metrics.strategicOrderingEvaluations++;
        const baseline = [...orderable].sort((left, right) =>
          left.total - right.total || left.baseScore - right.baseScore);
        const packingWeight = ["milestone", "detour"].includes(profile) ? 1 : 0.8;
        const doorwayWeight = payload.doorwayFlowWeight ?? 0.25;
        for (const candidate of orderable) {
          const doorway = typedDoorwayFlow(candidate.next.boxes, board);
          const packing = goalPackingBonus(candidate.next.boxes, board, {
            doorway,
            supportDependency: dependencyGraph,
            localAnalyses: localRooms,
            transition: candidate.next,
          });
          candidate.score += 0.2 * doorwayWeight * doorway.penalty - packingWeight * packing;
        }
        const enriched = [...orderable].sort((left, right) =>
          left.total - right.total || left.score - right.score);
        const changed = baseline.some((candidate, index) => candidate !== enriched[index]);
        if (changed) board.metrics.strategicOrderingChanges++;
        strategicOrderingGate.observe(changed);
      } else {
        board.metrics.strategicOrderingSkips++;
      }
    }
    candidates.sort((left, right) => left.total - right.total || left.score - right.score);
    for (const candidate of candidates) {
      if (candidate.total > threshold) {
        thresholdPrunes++;
        nextThresholdCandidate = Math.min(nextThresholdCandidate, candidate.total);
        nextThreshold = Math.min(nextThreshold, candidate.total);
        continue;
      }
      segments.push(candidate.next.path);
      const exceeded = searchContour(
        {robot: candidate.next.robot, boxes: candidate.next.boxes},
        candidate.cost,
        threshold,
        transpositions,
        accepted,
      );
      segments.pop();
      nextThreshold = Math.min(nextThreshold, exceeded);
      if (cutoff || solution) break;
    }
    activePath.delete(identity);
    return nextThreshold;
  };

  let threshold = bestEstimate;
  while (Number.isFinite(threshold) && threshold <= upperBound && !cutoff && !solution) {
    postMessage({type: "contour", threshold, visited, exactShard});
    activePath.clear();
    const transpositions = new BoundedDepthMap(payload.transpositionLimit || 80000);
    nextThresholdCandidate = Infinity;
    const nextThreshold = searchContour(initial, 0, threshold, transpositions);
    transpositionEvictions += transpositions.evictions;
    maxTranspositions = Math.max(maxTranspositions, transpositions.size);
    if (!Number.isFinite(nextThreshold)) break;
    if (nextThreshold <= threshold) threshold++;
    else threshold = nextThreshold;
  }
  return {
    path: solution,
    visited,
    cutoff,
    terminationReason: solution ? "solution" : cutoff ? "budget" : "bound-exhausted",
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
    transpositionEvictions,
    maxTranspositions,
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

function searchCore(payload) {
  if (payload.algorithm === "analyze-puzzle") {
    return {path: null, visited: 0, analysis: analyzePuzzleForSearch(payload.state)};
  }
  if (payload.algorithm === "bridge-astar") return bridgeAStarSearch(payload);
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
    if (goal(current.boxes, board.goals)) return {path: reconstructPath(cameFrom, identity), visited};
    if (pushMacro && createsSealedCorralDeadlock(current, board, reachable)) continue;
    if (payload.maxVisited && visited >= payload.maxVisited) {
      return {path: null, visited, cutoff: true};
    }
    const nextStates = pushMacro ? pushNeighbors(current, board, reachable)
      .map(next => expandPushMacro(next, board, payload.forcedMacros !== false))
      .filter(Boolean) :
      neighbors(current, board).map(n => ({robot: n.robot, boxes: n.boxes, path: [n.move]}));
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
    return {...result, performance: performanceSnapshot(metrics)};
  } finally {
    activePerformance = parentPerformance;
  }
}
