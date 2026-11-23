"""Brute-force check of the conditioning lemma on a 6-point set.
mu_K(T subset omega) = det K_T. Enumerate all 64 configurations, condition on A in, B out,
compare P(T subset omega | .) with det K'_T where K' = (I-(I-K)^<B>)^<A>, and check the chain
K' >= (K^<A>) restricted >= K^<A u B>."""
import itertools
import numpy as np

rng = np.random.default_rng(7)
n = 6
U, _ = np.linalg.qr(rng.standard_normal((n, n)))
K = U @ np.diag(rng.uniform(0.1, 0.9, n)) @ U.T
E = list(range(n))


def prob(S):
    # P(omega = S) = det(K_S - ... ) via inclusion-exclusion: det of (K on S, I-K off S) mixed matrix
    M = np.array([[K[i, j] if i in S else -K[i, j] for j in E] for i in E]) + np.diag([0 if i in S else 1 for i in E])
    return np.linalg.det(M)


def schur(Q, keep, drop):
    if not drop:
        return Q[np.ix_(keep, keep)]
    return Q[np.ix_(keep, keep)] - Q[np.ix_(keep, drop)] @ np.linalg.solve(Q[np.ix_(drop, drop)], Q[np.ix_(drop, keep)])


probs = {S: prob(set(S)) for r in range(n + 1) for S in itertools.combinations(E, r)}
print('total prob', round(sum(probs.values()), 12), 'min', min(probs.values()))
A, B = [0, 2], [4]
R = [x for x in E if x not in A and x not in B]
cond = {S: p for S, p in probs.items() if set(A) <= set(S) and not (set(B) & set(S))}
Z = sum(cond.values())
# kernel by the lemma's recipe
EB = [x for x in E if x not in B]
IminusK = np.eye(n) - K
KB = np.eye(len(EB)) - schur(IminusK, EB, B)  # on E\B
idxA = [EB.index(a) for a in A]
idxR = [EB.index(r) for r in R]
Kp = schur(KB, idxR, idxA)
err = 0.0
for r in range(1, len(R) + 1):
    for T in itertools.combinations(R, r):
        emp = sum(p for S, p in cond.items() if set(T) <= set(S)) / Z
        ii = [R.index(t) for t in T]
        err = max(err, abs(emp - np.linalg.det(Kp[np.ix_(ii, ii)])))
print('max |P(T|cond) - det K\'_T| =', err)
mid = schur(K, R, A)  # (K^<A>) restricted to R
low = schur(K, R, A + B)
print('min eig K\' - mid  :', np.linalg.eigvalsh(Kp - mid).min())
print('min eig mid - low  :', np.linalg.eigvalsh(mid - low).min())
print('eigs K\' - mid:', np.round(np.linalg.eigvalsh(Kp - mid), 6))
print('eigs mid - low:', np.round(np.linalg.eigvalsh(mid - low), 6))
