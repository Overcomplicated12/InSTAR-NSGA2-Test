from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import matplotlib.pyplot as plt

# --------------------------
# 1. Load a standard problem
# --------------------------
problem = get_problem("zdt1")

# --------------------------
# 2. Choose NSGA-II algorithm
# --------------------------
algorithm = NSGA2(pop_size=100)

# --------------------------
# 3. Termination condition
# --------------------------
termination = get_termination("n_gen", 50)

# --------------------------
# 4. Run optimization
# --------------------------
result = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)

# --------------------------
# 5. Print some results
# --------------------------
print("Optimization complete!")
print("First 10 solutions (objective values):")
print(result.F[:10])

# --------------------------
# 6. Plot the Pareto front
# --------------------------
F = result.F
plt.figure(figsize=(7,5))
plt.scatter(F[:,0], F[:,1], s=12)
plt.title("NSGA-II Pareto Front on ZDT1")
plt.xlabel("Objective f1")
plt.ylabel("Objective f2")
plt.grid(True)
plt.tight_layout()
plt.show()
