from typing import Dict, Set, Tuple, List


def compute_counts(ratio: Dict[str, float], total: int) -> Dict[str, int]:
    raw = {k: v * total / 100.0 for k, v in ratio.items()}
    cnt = {k: int(v) for k, v in raw.items()}
    rest = total - sum(cnt.values())
    leftovers = {k: raw[k] - cnt[k] for k in ratio}
    for k in sorted(leftovers, key=leftovers.get, reverse=True)[:rest]:
        cnt[k] += 1
    return cnt


def creates_forbidden_ngram(
    seq: List[int],
    cand: int,
    used: Set[Tuple[int, ...]],
    n: int,
) -> bool:
    if len(seq) + 1 < n:
        return False
    return tuple(seq[-(n - 1):] + [cand]) in used


def add_ngrams(
    seq: List[int],
    new: int,
    used: Set[Tuple[int, ...]],
    n: int,
) -> None:
    if len(seq) + 1 < n:
        return
    used.add(tuple(seq[-(n - 1):] + [new]))


def can_finish(
    remaining: Dict[str, int],
    cur: str,
    allowed: Dict[str, Set[str]],
    max_it: int = 500_000,
) -> bool:
    order = tuple(sorted(remaining))
    idx = {p: i for i, p in enumerate(order)}
    start = tuple(remaining[p] for p in order)
    stack = [(cur, start)]
    seen = set()
    it = 0

    while stack:
        node, state = stack.pop()
        it += 1
        if it > max_it:
            return False
        if (node, state) in seen:
            continue
        seen.add((node, state))
        if sum(state) == 0:
            return True

        remain = {p: state[idx[p]] for p in order}
        for nxt in allowed.get(node, ()):
            if remain.get(nxt, 0) > 0:
                nxt_state = list(state)
                nxt_state[idx[nxt]] -= 1
                stack.append((nxt, tuple(nxt_state)))

    return False