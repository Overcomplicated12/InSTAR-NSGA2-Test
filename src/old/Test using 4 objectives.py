import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import itertools

# ==========================================================
#  Vehicle Prediction + Smoothness Optimization Problem
# ==========================================================

class VehicleTrajectoryProblem(Problem):

    def __init__(self, T=50, dt=0.1):
        """
        T  = number of control steps
        dt = timestep
        """
        self.T = T
        self.dt = dt

        # Decision variables: throttle command at each timestep (range -2 to +2 m/s^2)
        super().__init__(n_var=T, n_obj=3, n_constr=0,
                         xl=-2.0, xu=2.0)

    # --------------------------------------------------
    #  Vehicle dynamics + objective calculations
    # --------------------------------------------------
    def _evaluate(self, X, out, *args, **kwargs):
        N = X.shape[0]   # number of population members
        T = self.T
        dt = self.dt

        # Objectives
        f_acc  = np.zeros(N)   # minimize acceleration magnitude
        f_smooth = np.zeros(N) # minimize jerk
        f_predict = np.zeros(N) # predictive penalty

        desired_speed = 20.0  # target speed (m/s)

        for i in range(N):
            a = X[i]                     # acceleration commands
            v = np.zeros(T+1)            # speed array
            jerk = np.diff(a) / dt       # jerk between steps

            # Integrate speed
            for t in range(T):
                v[t+1] = max(0, v[t] + a[t] * dt)

            # Objective 1: total absolute acceleration (energy)
            f_acc[i] = np.sum(np.abs(a))

            # Objective 2: smoothness = minimize jerk magnitude
            f_smooth[i] = np.sum(np.abs(jerk))

            # Objective 3: predictive cost = future deviation + future jerk
            # Predict 1s into the future using last known acceleration trend
            pred_horizon = 10  # 1 second (10 steps)
            future_v = v[-1]
            future_cost = 0
            for k in range(pred_horizon):
                future_acc = a[-1] if k == 0 else 0.9 * a[-1]  # predictive model
                future_v = max(0, future_v + future_acc * dt)
                future_cost += abs(future_v - desired_speed)

            # Add predicted jerk penalty
            future_cost += np.abs(jerk[-1]) * 5.0

            f_predict[i] = future_cost

        out["F"] = np.column_stack([f_acc, f_smooth, f_predict])


# ==========================================================
#  Run NSGA-II
# ==========================================================

problem = VehicleTrajectoryProblem(T=60, dt=0.1)

algorithm = NSGA2(pop_size=120)

termination = get_termination("n_gen", 60)

result = minimize(problem,
                  algorithm,
                  termination,
                  seed=1,
                  verbose=True)

F = result.F

print("Optimization complete!")
print("First few solutions:")
print(F[:5])

# ==========================================================
# Pairwise plot of the 3 objectives
# ==========================================================

pairs = list(itertools.combinations([0,1,2], 2))
plt.figure(figsize=(10, 7))

for i, (a, b) in enumerate(pairs):
    plt.subplot(2, 2, i+1)
    plt.scatter(F[:, a], F[:, b], s=15)
    plt.xlabel(f"Objective {a+1}")
    plt.ylabel(f"Objective {b+1}")
    plt.grid(True)
    plt.title(f"Obj {a+1} vs Obj {b+1}")

plt.tight_layout()
plt.show()

