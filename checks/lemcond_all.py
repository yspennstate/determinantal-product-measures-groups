"""Run the conditioning-lemma check of lemcond_check.py over every disjoint pair (A, B) on 6 points, 3 random kernels."""
import itertools
import numpy as np

def schur(Q, keep, drop):
    if not drop:
        return Q[np.ix_(keep, keep)]
    return Q[np.ix_(keep, keep)] - Q[np.ix_(keep, drop)] @ np.linalg.solve(Q[np.ix_(drop, drop)], Q[np.ix_(drop, keep)])

n = 6
E = list(range(n))
worst = [0.0, 0.0, 0.0]
cases = 0
for seed in (1, 2, 3):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    K = U @ np.diag(rng.uniform(0.05, 0.95, n)) @ U.T
    probs = {}
    for r in range(n + 1):
        for S in itertools.combinations(E, r):
            M = np.array([[K[i, j] if i in S else -K[i, j] for j in E] for i in E]) + np.diag([0 if i in S else 1 for i in E])
            probs[frozenset(S)] = np.linalg.det(M)
    for lab in itertools.product((0, 1, 2), repeat=n):  # 0 free, 1 in A, 2 in B
        A = [i for i in E if lab[i] == 1]
        B = [i for i in E if lab[i] == 2]
        R = [i for i in E if lab[i] == 0]
        if not R:
            continue
        cond = {S: p for S, p in probs.items() if set(A) <= S and not (set(B) & S)}
        Z = sum(cond.values())
        EB = [x for x in E if x not in B]
        KB = np.eye(len(EB)) - schur(np.eye(n) - K, EB, B)
        Kp = schur(KB, [EB.index(r) for r in R], [EB.index(a) for a in A])
        for r in range(1, len(R) + 1):
            for T in itertools.combinations(R, r):
                emp = sum(p for S, p in cond.items() if set(T) <= S) / Z
                ii = [R.index(t) for t in T]
                worst[0] = max(worst[0], abs(emp - np.linalg.det(Kp[np.ix_(ii, ii)])))
        mid = schur(K, R, A)
        low = schur(K, R, A + B)
        worst[1] = min(worst[1], np.linalg.eigvalsh(Kp - mid).min())
        worst[2] = min(worst[2], np.linalg.eigvalsh(mid - low).min())
        cases += 1
print('cases', cases, '| max kernel error', worst[0], '| min eig K\'-mid', worst[1], '| min eig mid-low', worst[2])
