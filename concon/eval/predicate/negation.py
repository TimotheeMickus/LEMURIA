#! This works (probably only) for unary predicates

import re
import pathlib
import multiprocessing as mp
import heapq
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

def _tokenize_messages(language: pd.DataFrame):
    assert "msg" in language.columns, "language must have a 'msg' column"
    tokenized = [[t for t in str(msg).split() if t != "0"] for msg in language["msg"]]
    vocab = sorted({t for toks in tokenized for t in toks})
    assert vocab, "vocab cannot be empty"
    return tokenized, vocab

def _build_presence_matrix(tokenized, vocab):
    '''Build binary matrix M of (|messages|, |vocab|).
    Set M[i,j]=1 if vocab[j] appears in message #i.'''
    M = np.zeros((len(tokenized), len(vocab)), dtype=np.uint8)
    tok2idx = {t: i for i, t in enumerate(vocab)}
    for i, toks in enumerate(tokenized):
        for t in toks:
            M[i, tok2idx[t]] = 1
    return M

def _parse_predicate(s):
    '''Parse a predicate into (value, negation bit).'''
    m = re.match(r"^\(¬(.+)\)$", str(s))
    if(m): return m.group(1), True
    return s, False

def _build_features(vocab, M):
    '''For each token t, define a binary feature vector over messages: X_t(S) = 1[t in s].'''
    features = {}
    for j, tok in enumerate(vocab):
        features[("tok", frozenset({tok}))] = M[:, j].astype(bool)
    return features

def _select_top_k(features, score_fn, top_k, score_cache=None):
    '''Keep the k highest-scoring features according to score_fn.'''
    if top_k is None or top_k >= len(features): return dict(features)
    if score_cache is None:
        return dict(sorted(features.items(), key=lambda kv: score_fn(kv[1]), reverse=True)[:top_k])
    return dict(sorted(features.items(), key=lambda kv: _score_cached(kv[0], kv[1], score_fn, score_cache), reverse=True)[:top_k])

def _score_cached(key, X, score_fn, score_cache):
    if key in score_cache: return score_cache[key]
    score_cache[key] = score_fn(X)
    return score_cache[key]

def _expand_features(features, parents, operator, score_fn, score_cache=None, max_new=None):
    '''Build composite features that strictly improve over both parent scores.
    If max_new is set, keep only top candidates by score during generation.'''
    assert operator in ["or", "and"]
    if(operator == "or"): op_fn = np.logical_or
    if(operator == "and"): op_fn = np.logical_and
    if(max_new is not None): max_new = int(max_new)
    items = list(parents.items())
    if(score_cache is None): parent_scores = {key: score_fn(X) for key, X in items}
    else: parent_scores = {key: _score_cached(key, X, score_fn, score_cache) for key, X in items}
    new_features, seen, order, top_heap = {}, set(), 0, []
    for i, (key_a, X_a) in enumerate(items):
        _, atoms_a = key_a
        for key_b, X_b in items[i+1:]:
            _, atoms_b = key_b
            if(atoms_a & atoms_b): continue
            atoms = atoms_a | atoms_b
            key = (operator, frozenset(atoms))
            if(key in features or key in seen): continue
            X = op_fn(X_a, X_b)
            if score_cache is None: score = score_fn(X)
            else:                   score = _score_cached(key, X, score_fn, score_cache)
            if(score <= parent_scores[key_a] or score <= parent_scores[key_b]): continue
            seen.add(key)
            if(max_new is None):
                new_features[key] = X
            else:
                rank = (float(score), -order)
                order += 1
                item = (rank, key, X)
                if(len(top_heap) < max_new): heapq.heappush(top_heap, item)
                elif(rank > top_heap[0][0]): heapq.heapreplace(top_heap, item)

    if(max_new is not None):
        top_heap.sort(key=lambda x: x[0], reverse=True)
        new_features = {key: X for _, key, X in top_heap}

    return new_features

def _grow_features(vocab, M, score_fn, operator,
    top_k=None, max_size=None, max_active_round=None, max_candidates=None, score_cache=None):
    '''Grow features by one atom per round up to max_size atoms.'''
    # Cache scores by feature key to avoid recomputing MI for the same candidate.
    if score_cache is None: score_cache = {}
    # First, score all single-token features
    features = _build_features(vocab, M)
    # Keep all unary candidates by construction; caps apply to growth only.
    if(max_size is not None and max_size <= 1): return features

    # Keep unary features available every round so growth is k -> k+1.
    unary = dict(features)
    # frontier are candidates created last round and parents for next expansion
    frontier = dict(features)
    current_size = 1
    best_score = max(_score_cached(k, X, score_fn, score_cache) for k, X in features.items())

    while True:
        # Avoid "fabricating" large, arbitrary features.
        if(max_size is not None and current_size >= max_size): break
        # Parent pruning - keep combinatorics tractable.
        round_top_k = None if current_size == 1 else top_k
        parents = _select_top_k(frontier, score_fn, top_k=round_top_k, score_cache=score_cache)
        # Mix current frontier with unary atoms to enable 1+2, 1+3, ...
        parents = {**unary, **parents}
        # Keep only the best candidates during generation to bound peak memory.
        cap_new = None
        if max_candidates is not None:
            remaining = int(max_candidates) - len(features)
            if(remaining <= 0): break
            cap_new = remaining
        if max_active_round is not None:
            cap_new = min(cap_new, int(max_active_round)) if cap_new is not None else int(max_active_round)
        if(cap_new is not None and cap_new <= 0): break
        new_features = _expand_features(
            features,
            parents,
            operator=operator,
            score_fn=score_fn,
            score_cache=score_cache,
            max_new=cap_new,
        )
        if max_size is not None:
            new_features = {k: X for k, X in new_features.items() if len(k[1]) <= max_size}
        if(not new_features): break
        # If feature size is unbounded, stop when the best new score no longer improves.
        # NB This is effectively not used/realistic in the current implementation
        round_best = max(_score_cached(k, X, score_fn, score_cache) for k, X in new_features.items())
        if(max_size is None and round_best <= best_score): break
        features.update(new_features)
        frontier = new_features
        best_score = max(best_score, round_best)
        current_size += 1

    return features

def _search_limits_from_vocab(vocab_size: int, profile: str = "fast"):
    """Bounds for search size by vocabulary size: (max_feat_size, top_k, max_active_round, max_candidates)."""
    v = int(vocab_size)
    p = str(profile).strip().lower()
    if p == "unbounded":
        # No hard caps on growth; stop only when no improving feature is found.
        return None, None, None, None
    if p == "slow":
        if(v >= 2000): return 4, 24, 1200, 24000
        if(v >= 800): return 5, 32, 1800, 36000
        if(v >= 300): return 6, 40, 2500, 50000
        if(v >= 120): return 7, 56, 3500, 70000
        return 8, 72, 5000, 100000

    if(v >= 1200): return 2, 10, 400, 4000
    if(v >= 400): return 3, 12, 600, 6000
    if(v >= 150): return 4, 14, 800, 8000
    return 5, 16, 1000, 10000

def _entropy(X):
    '''H(X) = -Σ_x p(x) log p(x).'''
    counts_x = Counter(X)
    n_x = sum(counts_x.values())
    return -sum(count / n_x * np.log2(count / n_x) for count in counts_x.values())

def _joint_entropy(*vars):
    '''H(X,Y) = -Σ_xy p(x,y) log p(x,y).'''
    assert len(vars) > 0, "need at least one variable"
    assert all(len(v) == len(vars[0]) for v in vars), "variables must be of equal length"
    return _entropy(list(zip(*vars)))

def _conditional_entropy(X, Y):
    '''H(X|Y) = -Σ_xy p(x,y) log p(x|y).'''
    assert len(X) == len(Y), "variables must be of equal length"
    counts_xy = Counter(zip(X,Y))
    counts_y = Counter(Y)
    n_xy = sum(counts_xy.values())
    return -sum(count / n_xy * np.log2(count / counts_y[xy[1]]) for xy, count in counts_xy.items())

def _mutual_information(X, Y):
    '''I(X;Y) = H(X) + H(Y) - H(X, Y).'''
    assert len(X) == len(Y), "variables must be of equal length"
    h_x = _entropy(X)
    h_y = _entropy(Y)
    h_xy = _joint_entropy(X,Y)
    return h_x + h_y - h_xy

def _conditional_mi(X, Y, Z):
    '''I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z).'''
    assert len(X) == len(Y) == len(Z), "variables must be of equal length"
    # Joint entropy
    h_xz = _joint_entropy(X,Z)
    h_yz = _joint_entropy(Y,Z)
    h_z = _entropy(Z)
    h_xyz = _joint_entropy(X,Y,Z)
    return h_xz + h_yz - h_z - h_xyz

def _binary_conditional_entropy(x1_counts, total_counts, n_total):
    '''H(X|C) for binary X and categorical C, from per-category counts of X=1 and totals.'''
    totals = np.asarray(total_counts, dtype=np.float64)
    if totals.size == 0 or n_total <= 0: return 0.0
    x1 = np.asarray(x1_counts, dtype=np.float64)
    mask = totals > 0
    if not np.any(mask): return 0.0
    p1 = x1[mask] / totals[mask]
    h = np.zeros_like(p1, dtype=np.float64)
    mid = (p1 > 0.0) & (p1 < 1.0)
    p = p1[mid]
    h[mid] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    weights = totals[mask] / float(n_total)
    return float(np.sum(weights * h))

def _cmi_binary_xn_given_v(X, v_codes, vn_codes, count_v, count_vn, n_total):
    '''I(X;N|V) for binary X, using H(X|V)-H(X|V,N).'''
    x = np.asarray(X, dtype=np.uint8)
    x1_by_v = np.bincount(v_codes, weights=x, minlength=len(count_v))
    x1_by_vn = np.bincount(vn_codes, weights=x, minlength=len(count_vn))
    return _binary_conditional_entropy(x1_by_v, count_v, n_total) - _binary_conditional_entropy(x1_by_vn, count_vn, n_total)

def _cmi_binary_xv_given_n(X, n_codes, vn_codes, count_n, count_vn, n_total):
    '''I(X;V|N) for binary X, using H(X|N)-H(X|V,N).'''
    x = np.asarray(X, dtype=np.uint8)
    x1_by_n = np.bincount(n_codes, weights=x, minlength=len(count_n))
    x1_by_vn = np.bincount(vn_codes, weights=x, minlength=len(count_vn))
    return _binary_conditional_entropy(x1_by_n, count_n, n_total) - _binary_conditional_entropy(x1_by_vn, count_vn, n_total)

def _safe_normalize(num, den):
    den = float(den)
    if(not np.isfinite(den) or den <= 0.0): return np.nan
    out = float(num) / den
    if(not np.isfinite(out)): return np.nan
    return out

def _safe_f1_2(a, b):
    """Two-term harmonic mean: 2ab/(a+b)."""
    if not (np.isfinite(a) and np.isfinite(b)): return np.nan
    den = a + b
    if den <= 0.0: return np.nan
    out = 2.0 * a * b / den
    return float(out) if np.isfinite(out) else np.nan

def _safe_f1_3(a, b, c):
    """Three-term harmonic mean: 3abc/(a+b+c)."""
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)): return np.nan
    den = a + b + c
    if den <= 0.0: return np.nan
    out = 3.0 * a * b * c / den
    return float(out) if np.isfinite(out) else np.nan

# S : set of all signals
# s : one observed signal
# X : candidate negation feature
# V : predicate value
# N : polarity bit (1 = negation, 0 = non-negation)
# If X = {t1, ..., tk}, then for any signal s,
# T = b(s) = (1[t1 ∈ s], ..., 1[tk ∈ s]) ∈ {0,1}^k
# is the activation pattern of X in s.
# R(s) : for a signal s, the remainder after removing the atoms of X
#        i.e. R(s) = s \ {t1, ..., tk} if X = {t1, ..., tk}
# n = I(X;N|V)/H(N|V) : X marque-t-il l'opposition de polarité relativement à V ?
# h = 1 - I(X;V|N)/H(V|N) : homogénéité (non-divulgation de la valeur du prédicat)
# t = I(T;V|X=1)/H(V|X=1) : value entanglement (fuite interne du feature composé)
# c = I(R;V|X=1)/H(V|X=1) : signal conservation après retrait de X (ancien u_T')

def _build_T(key, features):
    # key = (op, frozenset(atoms))
    _, atoms = key
    cols = [features[a] for a in sorted(atoms)]
    B = np.column_stack(cols) # shape: (n_msgs, |atoms|)
    return [tuple(row) for row in B]

def _build_R(key, tok_sets):
    # key = (op, frozenset(atoms))
    _, atoms = key
    return [msg_tokens.difference(atoms) for msg_tokens in tok_sets]

def _negation_strength(X, N, V):
    '''I(X;N|V) / H(N|V).
    Measures if a feature marks polarity relative to predicate value.'''
    return _safe_normalize(_conditional_mi(X, N, V), _conditional_entropy(N, V))

def _negation_strength_global(X, N):
    '''I(X;N) / H(N).
    Unconditioned polarity informativeness (global variant).'''
    return _safe_normalize(_mutual_information(X, N), _entropy(N))

def _homogeneity(X, V, N):
    '''1 - I(X;V|N) / H(V|N).
    Measures if the presence of a feature carries information on predicate value.'''
    return 1 - _safe_normalize(_conditional_mi(X, V, N), _conditional_entropy(V, N))

def _value_entanglement(T, V, X):
    '''I(T;V|X=1) / H(V|X=1).
    If the feature is composite, measures if its internal state carries information on predicate value.'''
    X = np.asarray(X).astype(bool)
    if(not np.any(X)): return np.nan
    T_0 = [t for t, keep in zip(T, X) if keep]
    V_0 = np.asarray(V)[X]
    h = _entropy(V_0)
    if(h <= 0): return np.nan
    return _mutual_information(T_0, V_0) / h

def _value_conservation(R, V, X):
    '''I(R;V|X=1) / H(V|X=1).
    After feature is removed from the signal, measures if its remainder carry information on predicate value.'''
    X = np.asarray(X, dtype=bool)
    if(not np.any(X)): return np.nan
    V_0 = np.asarray(V)[X]
    R_0 = [r for r, keep in zip(R, X) if keep]
    h = _entropy(V_0)
    if(h <= 0): return np.nan
    return _mutual_information(R_0, V_0) / h

def _orthogonality(X, R):
    '''I(X;R) in bits. Lower is more orthogonal.'''
    return _mutual_information(X, R)

def _value_conservation_global(R, V):
    '''Test variant: I(R;V) / H(V), without conditioning on X=1.'''
    h = _entropy(V)
    if(h <= 0): return np.nan
    return _mutual_information(R, V) / h

def _prefilter_size(n_features, top_rows, profile):
    if str(profile).strip().lower() in {"slow", "unbounded"}: return int(n_features)
    k = max(int(top_rows) * 20, 300)
    return min(int(n_features), k)

def analysis(language, sort_order: list = None, top_rows=10, profile: str = "fast",
    operator: str = "or"):
    assert operator in ["or", "and"], "operator must be 'or' or 'and'"
    # language: pd.DataFrame with "msg" and "pred_str" columns
    # V: np.ndarray shape (n_msgs,), predicate values (str/object)
    # N: np.ndarray shape (n_msgs,), negation flags (bool)
    V, N = map(np.array, zip(*language["pred_str"].apply(lambda s: _parse_predicate(s))))
    # tok: list[list[str]], length n_msgs
    # voc: list[str], length n_vocab
    tok, voc = _tokenize_messages(language)
    # Precompute token-sets used by _build_R.
    tok_sets = [frozenset(msg) for msg in tok]
    pred_labels = language["pred_str"].astype(str).to_numpy()
    # M: np.ndarray shape (n_msgs, n_vocab), binary token presence over messages
    M = _build_presence_matrix(tok, voc)
    # Entropy denominators do not depend on candidate feature X.
    h_n_given_v = _conditional_entropy(N, V)
    h_v_given_n = _conditional_entropy(V, N)
    h_n = _entropy(N)
    # Pre-encode conditioning variables for fast binary-X CMI.
    V_codes = pd.factorize(V, sort=False)[0].astype(np.int64, copy=False)
    N_codes = N.astype(np.int64, copy=False)
    VN_codes = V_codes * 2 + N_codes
    count_v = np.bincount(V_codes)
    count_n = np.bincount(N_codes, minlength=2)
    count_vn = np.bincount(VN_codes, minlength=max(1, 2 * len(count_v)))
    n_total = len(V_codes)
    # score_fn input: X -> np.ndarray shape (n_msgs,), bool/int
    # score_fn output: float
    score_fn = lambda X: _safe_normalize(_cmi_binary_xn_given_v(X, V_codes, VN_codes, count_v, count_vn, n_total), h_n_given_v)
    max_size, top_k, max_active_round, max_candidates = _search_limits_from_vocab(len(voc), profile=profile)
    score_cache = {}
    # _grow_features input: voc (n_vocab), M (n_msgs x n_vocab), ...
    # _grow_features output: dict[(op: str, atoms: frozenset[str]) -> X: np.ndarray shape (n_msgs,)]
    features = _grow_features(voc, M, score_fn, operator=operator, top_k=top_k, 
        max_size=max_size, max_active_round=max_active_round, max_candidates=max_candidates,
        score_cache=score_cache)
    # unary_features: dict[str -> np.ndarray shape (n_msgs,)]
    unary_features = {
        a: features[("tok", frozenset({a}))].astype(np.uint8) 
        for _, atoms in features.keys() if len(atoms) == 1 
        for a in atoms
        }

    # n determines what candidates we keep; we compute other metrics after
    cheap_rows = []
    for key, X in features.items():
        op, atoms = key
        cheap_rows.append({
            "_key": key,
            "_X": X,
            "feature": f"{op}({','.join(sorted(atoms))})",
            "n": _score_cached(key, X, score_fn, score_cache),
            "h": _safe_normalize(_cmi_binary_xv_given_n(X, N_codes, VN_codes, count_n, count_vn, n_total), h_v_given_n),
            "atoms": len(atoms),
            "support": float(np.mean(np.bincount(V_codes, weights=np.asarray(X, dtype=np.uint8), minlength=len(count_v)) > 0)),
        })

    # metrics for "survivors"
    prefilter_k = _prefilter_size(len(cheap_rows), top_rows=top_rows, profile=profile)
    def _prefilter_rank(row):
        n = row["n"] if np.isfinite(row["n"]) else -np.inf
        h = row["h"] if np.isfinite(row["h"]) else -np.inf
        return (n, h, row["support"], -row["atoms"])
    survivors = sorted(cheap_rows, key=_prefilter_rank, reverse=True)[:prefilter_k]

    rows = []
    for row in survivors:
        key, X = row["_key"], row["_X"]
        atom_count = row["atoms"]
        if(atom_count == 1): t = 0.0
        else:
            T = _build_T(key, unary_features)
            t = _value_entanglement(T, V, X)
        R = _build_R(key, tok_sets)
        c = _value_conservation(R, V, X)
        h = row["h"]
        o = _orthogonality(X, R)
        mask = np.asarray(X, dtype=bool)
        verified_predicates = pd.unique(pred_labels[mask]).tolist()
        rows.append({
            "feature": row["feature"],
            "n": row["n"],
            "n_global": _safe_normalize(_mutual_information(X, N), h_n),
            "h": h,
            "t": t,
            "c": c,
            "o": o,
            "F1_nr": _safe_f1_2(row["n"], c),
            "F1_nT": _safe_f1_2(row["n"], 1.0 - t),
            # Three-way harmonic mean over n, l_T, r (all high=good).
            "F1_nTr": _safe_f1_3(row["n"], 1.0 - t, c),
            "atoms": atom_count,
            "support": row["support"],
            "verified_predicates": " ; ".join(verified_predicates),
        })
    mapping = {"n": False, "n_global": False, "c": False, "o": False, "support": False, "atoms": True, "t": True, "h": False}
    if sort_order is not None:
        assert set(sort_order).issubset({"n", "n_global", "c", "o", "t", "h", "atoms", "support"}), "invalid key in sort_order"
    else:
        sort_order = ["n", "c", "t", "h"]
    ascending = [mapping[k] for k in sort_order]
    # pd.DataFrame shape (n_features, [feature, n, l_X, l_T, r, #atoms, support]), one row per feature
    report_df = pd.DataFrame(rows)
    report_df = report_df.sort_values(sort_order, ascending=ascending, na_position="last").head(top_rows)
    return report_df


def _analysis_report(task):
    language_df, top_rows, profile, operator, run_name, properties, epoch_number, vocab_size, eval_accuracy = task
    report = analysis(
        language_df,
        top_rows=top_rows,
        profile=profile,
        operator=operator,
    )
    report.insert(0, "run_name", run_name)
    report.insert(1, "operator", operator)
    report.insert(2, "properties", properties)
    report.insert(3, "epoch_analyzed", epoch_number)
    report.insert(4, "vocab_size", vocab_size)
    report.insert(5, "eval_accuracy", eval_accuracy)
    ordered_cols = [c for c in report.columns if c != "verified_predicates"] + ["verified_predicates"]
    report = report[ordered_cols]
    return report

def _analysis_report_indexed(args):
    idx, task = args
    return idx, _analysis_report(task)

def _best_accuracy_epoch(eval_df):
    """Return first epoch reaching max eval/accuracy, or None if unavailable."""
    if eval_df is None or eval_df.empty: return None
    if "eval/accuracy" not in eval_df.columns or "epoch" not in eval_df.columns: return None
    acc = pd.to_numeric(eval_df["eval/accuracy"], errors="coerce")
    ep = pd.to_numeric(eval_df["epoch"], errors="coerce")
    mask_valid = acc.notna() & ep.notna()
    if not mask_valid.any(): return None
    acc_valid = acc[mask_valid]
    ep_valid = ep[mask_valid]
    max_acc = float(acc_valid.max())
    # First epoch that reaches the maximum (with tiny tolerance for float noise).
    hit = np.isclose(acc_valid.to_numpy(dtype=np.float64), max_acc, rtol=0.0, atol=1e-12)
    if not np.any(hit): return None
    return int(ep_valid.iloc[np.argmax(hit)])

def _select_language_for_epoch(languages, target_epoch):
    """Select one language entry for the desired epoch, with fallback to closest available."""
    if not languages: return None
    if target_epoch is None:
        return max(languages, key=lambda x: x.get("epoch_number", -1))
    # Exact epoch match if available.
    exact = [x for x in languages if x.get("epoch_number", None) == target_epoch]
    if exact:
        return exact[0]
    # Fallback: closest dumped epoch to target; prefer earlier on ties.
    return min(
        languages,
        key=lambda x: (abs(int(x.get("epoch_number", -10**9)) - target_epoch), x.get("epoch_number", 10**9) > target_epoch),
    )

def export_analysis(datapoints_list, experiment_name: str, top_rows: int = 10,
    latest_only: bool = False, best_accuracy_only: bool = False, outputs_dir=None,
    profile: str = "fast", operator: str = "or", n_jobs: int = 1,
    filter_predicates: bool = False):
    """Perform analysis over datapoints and export a csv with top rows and a csv with the one top row."""
    if(outputs_dir is None): outputs_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    outputs_dir = pathlib.Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if latest_only and best_accuracy_only:
        raise ValueError("latest_only and best_accuracy_only are mutually exclusive.")

    all_top_rows = []
    top_one_rows = []
    n_jobs = max(1, int(n_jobs))

    # Build a task queue (show progress)
    tasks = []
    def _eval_acc_for_epoch(eval_df, epoch):
        if eval_df is None or eval_df.empty: return np.nan
        if "eval/accuracy" not in eval_df.columns: return np.nan
        if "epoch" in eval_df.columns:
            match = eval_df.loc[eval_df["epoch"] == epoch, "eval/accuracy"]
            if not match.empty: return float(match.iloc[-1])
        return np.nan

    for dp in datapoints_list:
        languages = dp["languages"]
        assert languages, "a datapoint is missing languages"

        if latest_only:
            language_entries = [max(languages, key=lambda x: x.get("epoch_number", -1))]
        elif best_accuracy_only:
            best_epoch = _best_accuracy_epoch(dp.get("evaluation"))
            language_entries = [_select_language_for_epoch(languages, best_epoch)]
        else:
            language_entries = sorted(languages, key=lambda x: x.get("epoch_number", -1))

        assert dp["evaluation"] is not None and not dp["evaluation"].empty, "a datapoint is missing evaluation dataframe"
        vocab_size = dp["evaluation"]["eval/vocab_used"].iloc[-1]
        for lang_entry in language_entries:
            epoch_number = lang_entry["epoch_number"]
            lang_df = lang_entry["language"]
            if filter_predicates:
                lang_df = lang_df[~lang_df["pred_str"].str.contains("∧", regex=False)]
            tasks.append((
                lang_df,
                top_rows,
                profile,
                operator,
                dp["run_name"],
                dp["config"]["properties"],
                epoch_number,
                vocab_size,
                    _eval_acc_for_epoch(dp["evaluation"], epoch_number),
            ))

    if n_jobs == 1:
        for task in tqdm(tasks, total=len(tasks), desc="Negation analysis"):
            report = _analysis_report(task)
            all_top_rows.append(report)
            top_one_rows.append(report.head(1))
    else:
        reports_by_idx = {}
        # maxtasksperchild=1 recycles workers after each task to hard-release memory.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs, maxtasksperchild=1) as pool:
            results = pool.imap_unordered(_analysis_report_indexed, enumerate(tasks), chunksize=1)
            with tqdm(total=len(tasks), desc=f"Negation analysis ({n_jobs} jobs)") as pbar:
                for idx, report in results:
                    reports_by_idx[idx] = report
                    pbar.update(1)
        for idx in sorted(reports_by_idx):
            report = reports_by_idx[idx]
            all_top_rows.append(report)
            top_one_rows.append(report.head(1))

    assert all_top_rows, "no rows collected for export"
    all_top_df = pd.concat(all_top_rows, ignore_index=True)
    top_one_df = pd.concat(top_one_rows, ignore_index=True)

    if latest_only:         mode_suffix = "_latest"
    elif best_accuracy_only: mode_suffix = "_best_accuracy"
    else:                   mode_suffix = ""
    all_top_path = outputs_dir / f"negation_top_rows_{experiment_name}_{operator}_{profile}{mode_suffix}.csv"
    top_one_path = outputs_dir / f"negation_top_1_{experiment_name}_{operator}_{profile}{mode_suffix}.csv"
    all_top_df.to_csv(all_top_path, index=False)
    top_one_df.to_csv(top_one_path, index=False)
    print(f"Saved negation top-rows CSV to: {all_top_path}")
    print(f"Saved negation top-1 CSV to: {top_one_path}")

    return all_top_df, top_one_df


if __name__ == "__main__":
    import pathlib
    import sys
    # for quick single-datapoint analysis; choose existing directory
    # language: pd.DataFrame with "msg" and "pred_str" columns
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    op = "or"
    feature_query = None
    if arg2 is not None:
        if arg2 in {"or", "and"}: op = arg2
        else:                     feature_query = arg2
    if arg3 is not None:
        if arg3 in {"or", "and"}: op = arg3
        elif feature_query is None: feature_query = arg3

    def _parse_feature_query(spec, default_operator="or"):
        s = str(spec).strip()
        m = re.fullmatch(r"(tok|or|and)\((.*)\)", s)
        if m:
            op_name = m.group(1)
            atoms = [a.strip() for a in m.group(2).split(",") if a.strip()]
            if op_name == "tok":
                if len(atoms) != 1: raise ValueError("tok(...) expects exactly one atom")
                return ("tok", frozenset({atoms[0]}))
            if not atoms: raise ValueError(f"{op_name}(...) expects at least one atom")
            return (op_name, frozenset(atoms))
        # Comma list without explicit operator -> use default operator.
        if "," in s:
            atoms = [a.strip() for a in s.split(",") if a.strip()]
            if not atoms: raise ValueError("empty feature specification")
            return (default_operator, frozenset(atoms))
        # Bare token -> unary token feature.
        return ("tok", frozenset({s}))

    def _feature_vector_from_key(key, tok_sets):
        op_name, atoms_f = key
        atoms_set = set(atoms_f)
        if op_name == "tok":
            atom = next(iter(atoms_set))
            return np.asarray([atom in msg_tokens for msg_tokens in tok_sets], dtype=np.uint8)
        if op_name == "or":
            return np.asarray([len(msg_tokens & atoms_set) > 0 for msg_tokens in tok_sets], dtype=np.uint8)
        if op_name == "and":
            return np.asarray([atoms_set.issubset(msg_tokens) for msg_tokens in tok_sets], dtype=np.uint8)
        raise ValueError(f"unknown operator: {op_name}")

    def _feature_label(key):
        op_name, atoms_f = key
        atoms_s = sorted(atoms_f)
        if op_name == "tok" and len(atoms_s) == 1: return f"tok({atoms_s[0]})"
        return f"{op_name}({','.join(atoms_s)})"

    experiment_input = sys.argv[1] if len(sys.argv) > 1 else input("Analyse: ").strip()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    runs_root = repo_root / "runs"
    raw_path = pathlib.Path(experiment_input)

    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        # If user gives a subpath (e.g. "interesting_data/file.csv"), treat it as explicit.
        if len(raw_path.parts) > 1:
            candidates.extend([pathlib.Path.cwd() / raw_path, runs_root / raw_path])
        else:
            # Bare filename -> default to runs/toyset/<file>.
            candidates.extend([runs_root / "toyset" / raw_path, pathlib.Path.cwd() / raw_path, runs_root / raw_path])

    input_path = next((p for p in candidates if p.is_file()), None)
    if input_path is None:
        tried = "\n".join(f"- {p}" for p in candidates)
        raise FileNotFoundError(f"Input file not found. Tried:\n{tried}")

    language = pd.read_csv(input_path)
    # display
    parsed = language["pred_str"].map(_parse_predicate)
    language_display = (language.assign(predicate=parsed.str[0], pol=np.where(parsed.str[1], "neg", "pos"))
        .pivot_table(index="predicate", columns="pol", values="msg", aggfunc=lambda s: " | ".join(pd.unique(s.astype(str))), fill_value="")
        .rename_axis(None, axis=1).reset_index().reindex(columns=["predicate", "pos", "neg"], fill_value=""))
    print(language_display.sort_values("predicate").to_string(index=False))
    # V: np.ndarray shape (n_msgs,), predicate values (str/object)
    # N: np.ndarray shape (n_msgs,), negation flags (bool)
    V, N = map(np.array, zip(*language["pred_str"].apply(lambda s: _parse_predicate(s))))
    V_codes = pd.factorize(V, sort=False)[0].astype(np.int64, copy=False)
    # tok: list[list[str]], length n_msgs
    # voc: list[str], length n_vocab
    tok, voc = _tokenize_messages(language)
    tok_sets = [frozenset(msg) for msg in tok]
    # M: np.ndarray shape (n_msgs, n_vocab), binary token presence over messages
    M = _build_presence_matrix(tok, voc)
    # score_fn input: X -> np.ndarray shape (n_msgs,), bool/int
    # score_fn output: float
    score_fn = lambda X: _negation_strength(X, N, V)
    # _grow_features input: voc (n_vocab), M (n_msgs x n_vocab), ...
    # _grow_features output: dict[(op: str, atoms: frozenset[str]) -> X: np.ndarray shape (n_msgs,)]
    features = None
    if feature_query is None:
        features = _grow_features(voc, M, score_fn, operator=op, top_k=None,
            max_size=len(voc), max_active_round=None, max_candidates=None)
    # unary_features: dict[str -> np.ndarray shape (n_msgs,)]
    unary_features = {voc[j]: M[:, j].astype(np.uint8) for j in range(len(voc))}

    def _row_for_feature(key, X):
        _, atoms = key
        T = _build_T(key, unary_features)
        R = _build_R(key, tok_sets)
        n_val = _negation_strength(X, N, V)
        h_val = _homogeneity(X, V, N)
        t_val = _value_entanglement(T, V, X)
        r_val = _value_conservation(R, V, X)
        return {
            "feature": _feature_label(key),
            "n": n_val,
            "n_global": _negation_strength_global(X, N),
            "h": h_val,
            "t": t_val,
            "c": r_val,
            "o": _orthogonality(X, R),
            "r_global": _value_conservation_global(R, V),
            "F1_nr": _safe_f1_2(n_val, r_val),
            "F1_nT": _safe_f1_2(n_val, 1.0 - t_val),
            "atoms": len(atoms),
            "support": float(np.mean(np.bincount(V_codes, weights=np.asarray(X, dtype=np.uint8), minlength=len(np.bincount(V_codes))) > 0)),
        }

    rows = []
    if feature_query is None:
        for key, X in features.items():
            rows.append(_row_for_feature(key, np.asarray(X, dtype=np.uint8)))
    else:
        key = _parse_feature_query(feature_query, default_operator=op)
        missing = [a for a in key[1] if a not in unary_features]
        if missing:
            print(f"[WARN] atoms not in vocabulary: {sorted(missing)} (treated as absent)")
            for a in missing:
                unary_features[a] = np.zeros(M.shape[0], dtype=np.uint8)
        X = _feature_vector_from_key(key, tok_sets)
        rows.append(_row_for_feature(key, X))

    # report: pd.DataFrame shape (n_features, [feature, n, l_X, l_T, r, #atoms, support]), one row per feature
    report = pd.DataFrame(rows).sort_values(["n", "c", "t", "h", "support"], ascending=[False, False, True, False, False], na_position="last")
    print(report.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
