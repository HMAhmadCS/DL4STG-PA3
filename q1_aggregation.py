"""Question 1: what information survives sum / mean / max aggregation."""
import itertools

import numpy as np

TRENDS = (-1, 0, 1)


def affine_sum(neigh, w, c):
    """Q1(a): sum of affine messages phi(x) = w x + c."""
    neigh = np.asarray(neigh)
    return w * neigh.sum() + c * len(neigh)


def q1a():
    print("Q1(a) affine message")
    A = [-1, 1]  # n=2, S=0, zero steady neighbours
    B = [0, 0]   # n=2, S=0, two steady neighbours
    for w, c in [(1.0, 0.0), (2.0, -3.0), (-0.7, 5.0)]:
        print(f"  w={w:+.1f} c={c:+.1f}: sum A={affine_sum(A, w, c):+.2f}, sum B={affine_sum(B, w, c):+.2f}")


def make_phi(K):
    """Q1(b): place-value message with base K+1 (no carries since every count <= K)."""
    base = K + 1
    return {-1: 1, 0: base, 1: base ** 2}, base


def decode(s, base):
    n_slow = s % base
    n_steady = (s // base) % base
    n_fast = s // base ** 2
    return n_slow, n_steady, n_fast


def q1b(K=8):
    print(f"Q1(b) injective scalar message, K={K}")
    phi, base = make_phi(K)
    print(f"  phi(-1)={phi[-1]}, phi(0)={phi[0]}, phi(+1)={phi[1]}")
    triples = [(a, b, c) for a in range(K + 1) for b in range(K + 1) for c in range(K + 1)
               if 1 <= a + b + c <= K]
    sums = {}
    for a, b, c in triples:
        neigh = [-1] * a + [0] * b + [1] * c
        s = sum(phi[x] for x in neigh)
        assert decode(s, base) == (a, b, c)
        sums.setdefault(s, []).append((a, b, c))
    assert all(len(v) == 1 for v in sums.values())
    print(f"  verified {len(triples)} count triples: all decoded correctly, {len(sums)} distinct sums")

    # Simpler target: phi(x) = x^2 sums to the number of non-steady neighbours.
    for n in range(1, K + 1):
        for neigh in itertools.product(TRENDS, repeat=n):
            neigh = np.array(neigh)
            assert (neigh ** 2).sum() == np.count_nonzero(neigh)
    print("  phi(x)=x^2 counts non-steady neighbours for every neighbourhood (verified n=1..8)")


ONE_HOT = {-1: np.array([1, 0, 0]), 0: np.array([0, 1, 0]), 1: np.array([0, 0, 1])}


def aggregate(neigh, method):
    m = np.stack([ONE_HOT[x] for x in neigh]).astype(float)
    return {"sum": m.sum(0), "mean": m.mean(0), "max": m.max(0)}[method]


def q1c(K=8):
    """Check which (task, aggregator) pairs are functions of the aggregated vector."""
    print("Q1(c) sum / mean / max with one-hot messages")
    tasks = {
        "count steady": lambda n: n.count(0),
        "majority slowing": lambda n: n.count(-1) > len(n) / 2,
        "any slowing": lambda n: -1 in n,
    }
    # Enumerate multisets (order does not matter for any aggregator).
    hoods = [list(h) for n in range(1, K + 1)
             for h in itertools.combinations_with_replacement(TRENDS, n)]
    for tname, task in tasks.items():
        for method in ("sum", "mean", "max"):
            seen = {}
            clash = None
            for h in hoods:
                key = tuple(round(float(v), 10) for v in aggregate(h, method))
                ans = task(h)
                if key in seen and seen[key][1] != ans:
                    clash = (seen[key][0], h, key)
                    break
                seen.setdefault(key, (h, ans))
            verdict = "Yes" if clash is None else "No "
            msg = "" if clash is None else f"  e.g. {clash[0]} vs {clash[1]} -> {clash[2]}"
            print(f"  {tname:17s} {method:4s}: {verdict}{msg}")

    # The counterexamples used in the report.
    for a, b in [([0], [0, 0]), ([-1, 0, 0], [-1, -1, 0])]:
        for method in ("mean", "max"):
            print(f"  {method}({a})={aggregate(a, method)}  {method}({b})={aggregate(b, method)}")


if __name__ == "__main__":
    q1a()
    q1b()
    q1c()
