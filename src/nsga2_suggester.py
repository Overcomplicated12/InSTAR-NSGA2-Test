import os
import csv, math, random
from dataclasses import dataclass
from typing import List, Dict, Tuple

# -----------------------
# CONFIG (edit these)
# -----------------------

# Which columns are "metrics" (objectives)
METRIC_COLS = ["runtime_ms", "veerScore", "lineLost"]
FINISH_COL = "finish"

# If finish == 0, add a big penalty so NSGA-II avoids non-finishers
FINISH_PENALTY_MS = 60_000

# How many new parameter sets to propose each run
NEXT_GEN_SIZE = 12

# Mutation settings
MUTATION_RATE = 0.25   # probability each gene mutates
MUTATION_SCALE = 0.12  # ~12% of range

# Parameter bounds (MUST match your Arduino safe ranges)
BOUNDS = {
    "kp": (0.05, 0.45),
    "kd": (0.2,  3.0),
    "base_speed": (120, 350),
    "min_base_speed": (0, 200),
    "corner1": (400, 1400),
    "corner2": (800, 2200),
    "corner3": (1200, 2600),
    "brake_pwr": (0, 40),
}

# Which parameter columns exist in your CSV (must be in BOUNDS)
PARAM_COLS = ["kp","kd","base_speed","min_base_speed","corner1","corner2","corner3","brake_pwr"]

# Objective weights (bigger = more important)
OBJ_WEIGHTS = {
    "runtime_ms": 5.0,
    "veerScore":  1.0,
    "lineLost":   3.0,   # example: punish losing the line more
}

# If you want weights to mean "importance" but keep magnitudes comparable:
# set NORMALIZE_OBJECTIVES = True and provide rough scales.
NORMALIZE_OBJECTIVES = True
OBJ_SCALES = {
    "runtime_ms": 10000.0,  # typical runtime range
    "veerScore":  20.0,     # typical veer range
    "lineLost":   50.0,     # typical lineLost range
}

# Setting up paths to the data file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIALS_PATH = os.path.join(BASE_DIR, "..", "data", "trials.csv")


# -----------------------
# NSGA-II core
# -----------------------

@dataclass
class Trial:
    params: Dict[str, float]
    metrics: Dict[str, float]
    # computed objectives list (same order as METRIC_COLS)
    obj: Tuple[float, float, float] = None

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def read_trials(path: str) -> List[Trial]:
    out = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            params = {}
            for p in PARAM_COLS:
                if p not in row:
                    raise ValueError(f"Missing param column '{p}' in CSV.")
                params[p] = float(row[p])

            metrics = {}
            for m in METRIC_COLS + [FINISH_COL]:
                if m not in row:
                    raise ValueError(f"Missing metric column '{m}' in CSV.")
                metrics[m] = float(row[m])

            t = Trial(params=params, metrics=metrics)
            out.append(t)
    return out

def compute_objectives(t: Trial) -> Tuple[float, float, float]:
    # Objectives are minimized
    runtime = t.metrics["runtime_ms"]
    if int(t.metrics[FINISH_COL]) == 0:
        runtime += FINISH_PENALTY_MS

    veer = t.metrics["veerScore"]
    lost = t.metrics["lineLost"]

    if NORMALIZE_OBJECTIVES:
        runtime_n = runtime / max(OBJ_SCALES["runtime_ms"], 1e-9)
        veer_n    = veer    / max(OBJ_SCALES["veerScore"], 1e-9)
        lost_n    = lost    / max(OBJ_SCALES["lineLost"], 1e-9)
    else:
        runtime_n, veer_n, lost_n = runtime, veer, lost

    runtime_w = runtime_n * OBJ_WEIGHTS["runtime_ms"]
    veer_w    = veer_n    * OBJ_WEIGHTS["veerScore"]
    lost_w    = lost_n    * OBJ_WEIGHTS["lineLost"]

    # Tiny tiebreaker favoring faster runs (safe for NSGA-II)
    runtime_w -= 1e-6 * runtime_n

    return (runtime_w, veer_w, lost_w)


def dominates(a: Tuple[float,...], b: Tuple[float,...]) -> bool:
    # a dominates b if a is <= b in all objectives and < in at least one
    le_all = all(x <= y for x, y in zip(a, b))
    lt_any = any(x < y for x, y in zip(a, b))
    return le_all and lt_any

def fast_nondominated_sort(pop: List[Trial]) -> List[List[int]]:
    # Returns list of fronts, each front is list of indices
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    fronts: List[List[int]] = [[]]

    for i in range(len(pop)):
        for j in range(len(pop)):
            if i == j:
                continue
            if dominates(pop[i].obj, pop[j].obj):
                S[i].add(j)
            elif dominates(pop[j].obj, pop[i].obj):
                n[i] += 1
        if n[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    fronts.pop()  # last empty
    return fronts


def choose_single_solution(pareto_indices: List[int], trials: List[Trial]) -> Trial:
    """
    Choose ONE solution from the Pareto front using explicit priorities.
    Priority order:
      1) finish == 1
      2) lowest runtime_ms
      3) lowest brake_pwr
      4) lowest veerScore
      5) lowest lineLost
    """

    pareto_trials = [trials[i] for i in pareto_indices]

    # 1) keep only finishers if possible
    finishers = [t for t in pareto_trials if int(t.metrics["finish"]) == 1]
    if finishers:
        pareto_trials = finishers

    # 2) sort by explicit priority
    pareto_trials.sort(
        key=lambda t: (
            t.metrics["runtime_ms"],
            t.params["brake_pwr"],
            t.metrics["veerScore"],
            t.metrics["lineLost"],
        )
    )

    return pareto_trials[0]


def crowding_distance(front: List[int], pop: List[Trial]) -> Dict[int, float]:
    dist = {i: 0.0 for i in front}
    num_obj = len(pop[0].obj)

    for m in range(num_obj):
        front_sorted = sorted(front, key=lambda i: pop[i].obj[m])
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")

        fmin = pop[front_sorted[0]].obj[m]
        fmax = pop[front_sorted[-1]].obj[m]
        if fmax == fmin:
            continue

        for k in range(1, len(front_sorted)-1):
            prevv = pop[front_sorted[k-1]].obj[m]
            nextv = pop[front_sorted[k+1]].obj[m]
            dist[front_sorted[k]] += (nextv - prevv) / (fmax - fmin)

    return dist

def nsga2_select(pop: List[Trial], k: int) -> List[Trial]:
    fronts = fast_nondominated_sort(pop)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend([pop[i] for i in front])
        else:
            cd = crowding_distance(front, pop)
            front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
            needed = k - len(selected)
            selected.extend([pop[i] for i in front_sorted[:needed]])
            break
    return selected

# -----------------------
# Variation operators
# -----------------------

def crossover(p1: Dict[str,float], p2: Dict[str,float]) -> Dict[str,float]:
    # Blend crossover for floats/ints
    child = {}
    for key in PARAM_COLS:
        lo, hi = BOUNDS[key]
        a = p1[key]
        b = p2[key]
        alpha = random.random()
        v = alpha*a + (1-alpha)*b
        child[key] = clamp(v, lo, hi)
    # keep int-like params as ints
    for key in ["base_speed","min_base_speed","corner1","corner2","corner3","brake_pwr"]:
        child[key] = int(round(child[key]))
    return child

def mutate(p: Dict[str,float]) -> Dict[str,float]:
    out = dict(p)
    for key in PARAM_COLS:
        if random.random() < MUTATION_RATE:
            lo, hi = BOUNDS[key]
            span = hi - lo
            delta = random.gauss(0, MUTATION_SCALE*span)
            out[key] = clamp(out[key] + delta, lo, hi)
    for key in ["base_speed","min_base_speed","corner1","corner2","corner3","brake_pwr"]:
        out[key] = int(round(out[key]))
    return out

def make_child(parents: List[Trial]) -> Dict[str,float]:
    p1, p2 = random.sample(parents, 2)
    c = crossover(p1.params, p2.params)
    c = mutate(c)
    return c

def unique_params(cands: List[Dict[str,float]], seen: set) -> List[Dict[str,float]]:
    out = []
    for c in cands:
        key = tuple(c[p] for p in PARAM_COLS)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

# -----------------------
# Main: suggest next gen
# -----------------------

def main():
    random.seed()  # system randomness

    trials = read_trials(TRIALS_PATH)
    if len(trials) < 6:
        print("Add at least ~6 trials for NSGA-II to have something to work with.")
        return

    # attach objectives
    for t in trials:
        t.obj = compute_objectives(t)

    # Select a parent pool (top half)
    parent_pool_size = max(4, len(trials)//2)
    parents = nsga2_select(trials, parent_pool_size)

    # Build new candidates
    seen = set(tuple(t.params[p] for p in PARAM_COLS) for t in trials)
    children = []
    while len(children) < NEXT_GEN_SIZE:
        c = make_child(parents)
        key = tuple(c[p] for p in PARAM_COLS)
        if key not in seen:
            seen.add(key)
            children.append(c)

    # Print suggested next generation
    print("SUGGESTED_NEXT_GEN")
    print(",".join(PARAM_COLS))
    for c in children:
        print(",".join(str(c[p]) for p in PARAM_COLS))

    # Also show current Pareto front (best tradeoffs)
    fronts = fast_nondominated_sort(trials)
    pareto = fronts[0]

    best = choose_single_solution(pareto, trials)

    print("\nCHOSEN_BEST_SOLUTION")
    print(",".join(PARAM_COLS))
    print(",".join(str(best.params[p]) for p in PARAM_COLS))

    print("\nBEST_SOLUTION_METRICS")
    print("runtime_ms,veerScore,lineLost,finish")
    print(
        f"{best.metrics['runtime_ms']},"
        f"{best.metrics['veerScore']},"
        f"{best.metrics['lineLost']},"
        f"{int(best.metrics['finish'])}"
    )


if __name__ == "__main__":
    main()
