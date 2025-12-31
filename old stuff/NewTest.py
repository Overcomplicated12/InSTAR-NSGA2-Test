import random, time, re
import serial

PORT = "COM5"       # change this
BAUD = 115200
POP = 24
GENS = 20
TRIAL_TIMEOUT = 20  # seconds (trial is ~8s + overhead)

# --- Parameter ranges (adjust if needed) ---
R_KP = (0.05, 0.60)
R_KD = (0.10, 4.00)
R_BASE = (90, 280)
R_MINB = (60, 180)

def clamp(x, lo, hi): return max(lo, min(hi, x))

def random_individual():
    kp = random.uniform(*R_KP)
    kd = random.uniform(*R_KD)
    base = random.randint(*R_BASE)
    minb = random.randint(*R_MINB)
    minb = clamp(minb, 40, base)
    return [kp, kd, base, minb]

def mutate(ind, pm=0.25):
    kp, kd, base, minb = ind
    if random.random() < pm: kp += random.uniform(-0.05, 0.05)
    if random.random() < pm: kd += random.uniform(-0.30, 0.30)
    if random.random() < pm: base += random.randint(-20, 20)
    if random.random() < pm: minb += random.randint(-20, 20)
    kp = clamp(kp, *R_KP)
    kd = clamp(kd, *R_KD)
    base = int(clamp(base, *R_BASE))
    minb = int(clamp(minb, 40, base))
    return [kp, kd, base, minb]

def crossover(a, b):
    # simple blend crossover
    out = []
    for i in range(4):
        if i < 2:
            t = random.random()
            out.append(a[i]*t + b[i]*(1-t))
        else:
            out.append(int(round((a[i] + b[i]) / 2)))
    out[2] = int(clamp(out[2], *R_BASE))
    out[3] = int(clamp(out[3], 40, out[2]))
    return out

RESULT_RE = re.compile(r"TRIAL_RESULT\s+meanAbsErr=([0-9.]+)\s+maxAbsErr=([0-9]+)\s+speedProxy=([0-9.]+)\s+lost=([01])")

def evaluate(ser, ind):
    kp, kd, base, minb = ind
    cmd = f"P {kp:.4f} {kd:.4f} {base:d} {minb:d}\n".encode()
    ser.write(cmd)
    ser.flush()

    # You must press the button on the robot to start each trial
    # OR: you can modify Arduino code to auto-run on receiving 'GO'
    print(f"Set params: KP={kp:.4f} KD={kd:.4f} BASE={base} MINB={minb}")
    print("Now press the Zumo button to run a trial...")

    t0 = time.time()
    while time.time() - t0 < TRIAL_TIMEOUT:
        line = ser.readline().decode(errors="ignore").strip()
        m = RESULT_RE.search(line)
        if m:
            meanAbsErr = float(m.group(1))
            maxAbsErr = int(m.group(2))
            speedProxy = float(m.group(3))
            lost = int(m.group(4))

            # Objectives to MINIMIZE:
            # 1) meanAbsErr
            # 2) maxAbsErr
            # 3) -speedProxy (so faster is better)
            # 4) lost penalty
            lostPenalty = 5000.0 if lost else 0.0
            return (meanAbsErr + lostPenalty, maxAbsErr + int(lostPenalty), -speedProxy, lost)
    # Timeout => treat as bad
    return (999999.0, 999999, 0.0, 1)

def dominates(f1, f2):
    # Pareto dominance for minimization
    return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

def fast_nondominated_sort(pop, फिट):
    fronts = []
    S = {i: [] for i in range(len(pop))}
    n = {i: 0 for i in range(len(pop))}
    rank = {}

    F0 = []
    for i in range(len(pop)):
        for j in range(len(pop)):
            if i == j: continue
            if dominates(फिट[i], फिट[j]):
                S[i].append(j)
            elif dominates(फिट[j], फिट[i]):
                n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            F0.append(i)

    fronts.append(F0)
    k = 0
    while fronts[k]:
        Q = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    rank[j] = k + 1
                    Q.append(j)
        k += 1
        fronts.append(Q)
    return fronts[:-1]

def crowding_distance(front, फिट):
    if not front: return {}
    dist = {i: 0.0 for i in front}
    M = len(फिट[0])
    for m in range(M):
        front_sorted = sorted(front, key=lambda i: फिट[i][m])
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")
        fmin = फिट[front_sorted[0]][m]
        fmax = फिट[front_sorted[-1]][m]
        if fmax == fmin: continue
        for k in range(1, len(front_sorted) - 1):
            prevv = फिट[front_sorted[k - 1]][m]
            nextv = फिट[front_sorted[k + 1]][m]
            dist[front_sorted[k]] += (nextv - prevv) / (fmax - fmin)
    return dist

def select_nsga2(pop, फिट, N):
    fronts = fast_nondominated_sort(pop, फिट)
    new = []
    for front in fronts:
        if len(new) + len(front) <= N:
            new.extend(front)
        else:
            cd = crowding_distance(front, फिट)
            front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
            new.extend(front_sorted[: N - len(new)])
            break
    return [pop[i] for i in new], [फिट[i] for i in new]

def tournament(pop, फिट, k=2):
    best = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best is None or dominates(फिट[i], फिट[best]) or (not dominates(फिट[best], फिट[i]) and random.random() < 0.5):
            best = i
    return pop[best]

def main():
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        time.sleep(2)
        ser.reset_input_buffer()

        population = [random_individual() for _ in range(POP)]

        fitness = []
        for ind in population:
            fitness.append(evaluate(ser, ind))

        for g in range(GENS):
            print(f"\n=== Generation {g+1}/{GENS} ===")
            offspring = []
            while len(offspring) < POP:
                p1 = tournament(population, fitness)
                p2 = tournament(population, fitness)
                child = crossover(p1, p2)
                child = mutate(child)
                offspring.append(child)

            off_fit = [evaluate(ser, ind) for ind in offspring]

            combined = population + offspring
            combined_fit = fitness + off_fit

            population, fitness = select_nsga2(combined, combined_fit, POP)

            # Print best Pareto front
            fronts = fast_nondominated_sort(population, fitness)
            F0 = fronts[0]
            print("Pareto front (first 5):")
            for i in F0[:5]:
                print("  ind=", population[i], "fit=", fitness[i])

        # Final Pareto front
        fronts = fast_nondominated_sort(population, fitness)
        print("\n=== FINAL PARETO FRONT ===")
        for i in fronts[0]:
            print("ind=", population[i], "fit=", fitness[i])

if __name__ == "__main__":
    main()
