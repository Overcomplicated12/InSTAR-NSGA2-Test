import os
import csv
import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# -----------------------
# CONFIG (edit these)
# -----------------------

METRIC_COLS = ["runtime_ms", "veerScore", "lineLost"]
FINISH_COL = "finish"
FINISH_PENALTY_MS = 60_000

NEXT_GEN_SIZE = 12
MUTATION_PROB = 0.25
MUTATION_ETA = 20
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 15

# Parameter bounds (MUST match Arduino safe ranges)
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

PARAM_COLS = ["kp","kd","base_speed","min_base_speed","corner1","corner2","corner3","brake_pwr"]

# Objective weights (bigger = more important)
OBJ_WEIGHTS = {
    "runtime_ms": 5.0,
    "veerScore":  1.0,
    "lineLost":   3.0,
}

# Optional normalization so weights behave predictably
NORMALIZE_OBJECTIVES = True
OBJ_SCALES = {
    "runtime_ms": 10000.0,
    "veerScore":  20.0,
    "lineLost":   50.0,
}

# Path to trials.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIALS_PATH = os.path.join(BASE_DIR, "trials.csv")  # put trials.csv next to this script


# -----------------------
# Helpers
# -----------------------

INT_PARAMS = {"base_speed","min_base_speed","corner1","corner2","corner3","brake_pwr"}

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def read_trials_csv(path):
    X = []
    M = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # params
            x = []
            for p in PARAM_COLS:
                if p not in row:
                    raise ValueError(f"Missing param column '{p}' in CSV header.")
                x.append(float(row[p]))
            # metrics
            met = {}
            for m in METRIC_COLS + [FINISH_COL]:
                if m not in row:
                    raise ValueError(f"Missing metric column '{m}' in CSV header.")
                met[m] = float(row[m])
            X.append(x)
            M.append(met)
    return np.array(X, dtype=float), M

def compute_F(metrics_list):
    F = []
    for met in metrics_list:
        runtime = met["runtime_ms"]
        if int(met[FINISH_COL]) == 0:
            runtime += FINISH_PENALTY_MS

        veer = met["veerScore"]
        lost = met["lineLost"]

        if NORMALIZE_OBJECTIVES:
            runtime_n = runtime / max(OBJ_SCALES["runtime_ms"], 1e-9)
            veer_n    = veer    / max(OBJ_SCALES["veerScore"], 1e-9)
            lost_n    = lost    / max(OBJ_SCALES["lineLost"], 1e-9)
        else:
            runtime_n, veer_n, lost_n = runtime, veer, lost

        runtime_w = runtime_n * OBJ_WEIGHTS["runtime_ms"]
        veer_w    = veer_n    * OBJ_WEIGHTS["veerScore"]
        lost_w    = lost_n    * OBJ_WEIGHTS["lineLost"]

        # tiny tie-break toward faster
        runtime_w -= 1e-6 * runtime_n

        F.append([runtime_w, veer_w, lost_w])
    return np.array(F, dtype=float)

def round_and_clip(X):
    X2 = X.copy()
    for j, name in enumerate(PARAM_COLS):
        lo, hi = BOUNDS[name]
        X2[:, j] = np.clip(X2[:, j], lo, hi)
        if name in INT_PARAMS:
            X2[:, j] = np.rint(X2[:, j])
    return X2

def key_tuple(xrow):
    # stable tuple for de-dup (ints stay ints)
    out = []
    for v, name in zip(xrow, PARAM_COLS):
        if name in INT_PARAMS:
            out.append(int(v))
        else:
            out.append(round(float(v), 12))
    return tuple(out)


# -----------------------
# Minimal Problem (only bounds needed for operators)
# -----------------------

class ParamProblem(Problem):
    def __init__(self):
        xl = np.array([BOUNDS[p][0] for p in PARAM_COLS], dtype=float)
        xu = np.array([BOUNDS[p][1] for p in PARAM_COLS], dtype=float)
        super().__init__(n_var=len(PARAM_COLS), n_obj=3, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        raise RuntimeError("We do not evaluate with pymoo here; objectives come from your trials.csv.")


# -----------------------
# Main
# -----------------------

def main():
    np.random.seed(None)

    X, metrics_list = read_trials_csv(TRIALS_PATH)
    if len(X) < 6:
        print("Add at least ~6 trials for NSGA-II style selection to work well.")
        return

    F = compute_F(metrics_list)

    problem = ParamProblem()

    # Create a population from your past trials
    pop = Population.new("X", X, "F", F)

    # Pick parent pool (top ~half) using Rank+Crowding (NSGA-II survival)
    parent_pool_size = max(4, len(pop) // 2)
    survival = RankAndCrowding()
    parents_pop = survival.do(problem, pop, n_survive=parent_pool_size)

    # Operators
    cx = SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
    mut = PM(prob=MUTATION_PROB, eta=MUTATION_ETA)

    # Generate offspring
    seen = set(key_tuple(row) for row in X)
    children = []

    # We'll keep trying until we have enough unique kids
    while len(children) < NEXT_GEN_SIZE:
        n_matings = max(1, NEXT_GEN_SIZE - len(children))

        # parent index pairs into parents_pop
        P = np.column_stack([
            np.random.randint(0, len(parents_pop), size=n_matings),
            np.random.randint(0, len(parents_pop), size=n_matings),
        ])

        off = cx.do(problem, parents_pop, P)     # crossover
        off = mut.do(problem, off)              # mutation

        Xoff = off.get("X")
        Xoff = round_and_clip(Xoff)

        for row in Xoff:
            k = key_tuple(row)
            if k not in seen:
                seen.add(k)
                children.append(row)
            if len(children) >= NEXT_GEN_SIZE:
                break

    children = np.array(children, dtype=float)

    # Print suggested next gen (paste into Arduino)
    print("SUGGESTED_NEXT_GEN")
    print(",".join(PARAM_COLS))
    for row in children:
        out = []
        for v, name in zip(row, PARAM_COLS):
            if name in INT_PARAMS:
                out.append(str(int(v)))
            else:
                out.append(f"{float(v):.12g}")
        print(",".join(out))

    # Print Pareto front from existing trials (based on weighted objectives F)
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    print("\nCURRENT_PARETO_FRONT (based on weighted objectives)")
    print("runtime_ms,veerScore,lineLost,finish," + ",".join(PARAM_COLS))
    for idx in nds:
        met = metrics_list[idx]
        row = X[idx]
        print(f"{met['runtime_ms']},{met['veerScore']},{met['lineLost']},{int(met[FINISH_COL])}," +
              ",".join(str(int(v)) if name in INT_PARAMS else f"{float(v):.12g}"
                       for v, name in zip(row, PARAM_COLS)))


if __name__ == "__main__":
    main()
