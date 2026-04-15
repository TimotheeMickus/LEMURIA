import re
import pandas as pd
import numpy as np
from collections import Counter

def _tokenize_messages(language: pd.DataFrame):
    assert "msg" in language.columns, "language must have a 'msg' column"
    tokenized = [[t for t in str(msg).split() if t != "0"] for msg in language["msg"]]
    vocab = sorted({t for toks in tokenized for t in toks})
    assert vocab, "vocab cannot be empty"
    return tokenized, vocab

def _build_presence_matrix(tokenized, vocab):
    '''Build binary matrix M of (|messages|, |vocab|).
    Set M[i,j]=1 if vocab[j] appears in message #i.'''
    M = np.zeros((len(tokenized), len(vocab)), dtype=int)
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
    return {("tok", frozenset({tok})): M[:, j].astype(bool) for j, tok in enumerate(vocab)}

def _select_top_k(features, score_fn, top_k):
    '''Keep the k highest-scoring features according to score_fn.'''
    return dict(sorted(features.items(), key=lambda kv: score_fn(kv[1]), reverse=True)[:top_k])

def _expand_features(features, parents, operator, score_fn):
    '''Build composite features that strictly improve over both parent scores.'''
    assert operator in ["or", "and"]
    if(operator == "or"): op_fn = np.logical_or
    if(operator == "and"): op_fn = np.logical_and
    items = list(parents.items())
    parent_scores = {key: score_fn(X) for key, X in items}
    new_features = {}
    for i, (key_a, X_a) in enumerate(items):
        _, atoms_a = key_a
        for key_b, X_b in items[i+1:]:
            _, atoms_b = key_b
            if(atoms_a & atoms_b): continue
            atoms = atoms_a | atoms_b
            key = (operator, frozenset(atoms))
            if(key in features or key in new_features): continue
            X = op_fn(X_a, X_b)
            score = score_fn(X)
            if(score <= parent_scores[key_a] or score <= parent_scores[key_b]): continue
            new_features[key] = X
    return new_features

def _grow_features(vocab, M, score_fn, operator, top_k=None, max_size=None):
    '''Grow features over max_size rounds or as long as score improves.'''
    features = _build_features(vocab, M)
    if(max_size is not None and max_size <= 1):
        return features

    frontier = dict(features)
    current_size = 1
    best_score = max(score_fn(X) for X in features.values())

    while True:
        if(max_size is not None and current_size >= max_size): break
        round_top_k = None if current_size == 1 else top_k
        parents = _select_top_k(frontier, score_fn, top_k=round_top_k)
        new_features = _expand_features(features, parents, operator=operator, score_fn=score_fn)
        if(not new_features): break
        round_best = max(score_fn(X) for X in new_features.values())
        if(max_size is None and round_best <= best_score): break
        features.update(new_features)
        frontier = new_features
        best_score = max(best_score, round_best)
        current_size += 1

    return features

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

def _conditional_mi(X, Y, Z):
    '''I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z).'''
    assert len(X) == len(Y) == len(Z), "variables must be of equal length"
    # Joint entropy
    h_xz = _joint_entropy(X,Z)
    h_yz = _joint_entropy(Y,Z)
    h_z = _entropy(Z)
    h_xyz = _joint_entropy(X,Y,Z)
    return h_xz + h_yz - h_z - h_xyz

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
# l_X = I(X;V|N)/H(V|N) : la présence de X divulgue-t-elle de l'information sur la valeur du prédicat ?
# l_T = I(T;V|X=1)/H(V|X=1) : si X est composé, son état interne divulgue-t-il la valeur du prédicat ?
# r = I(R;V|X=1)/H(V|X=1) : après retrait de X, le reste du signal R conserve-t-il la valeur du prédicat ? (ancien u_T')

def _negation_strength(X, N, V):
    '''I(X;N|V) / H(N|V).
    Measures if a feature marks polarity relative to predicate value.'''
    return _conditional_mi(X, N, V) / _conditional_entropy(N, V)

def _leak_unary(X, V, N):
    '''I(X;V|N) / H(V|N).
    Measures if the presence of a feature carries information on predicate value.'''
    return _conditional_mi(X, V, N) / _conditional_entropy(V, N)

def _leak_composite(T, V, X):
    '''I(T;V|X=1) / H(V|X=1).
    If the feature is composite, measures if its internal state carries information on predicate value.'''
    assert True, "TODO"
    pass

def _signal_conservation(R, V, X):
    '''I(R;V|X=1) / H(V|X=1).
    After feature is removed from the signal, measures if its remainder carry information on predicate value.'''
    pass


# DEBUG
if __name__ == "__main__":
    import pathlib
    directory = pathlib.Path(__file__).resolve().parents[3] / "runs" / "n_complexity_memory" / "props=16__d=1-2__cand=2__enc=node_averager__cs=balanced__t=2026-03-27_12-40-18__run=0"
    filename = "msgs.e7.csv"
    language = pd.read_csv(directory / filename)
    V, N = map(np.array, zip(*language["pred_str"].apply(lambda s: _parse_predicate(s))))
    tok, voc = _tokenize_messages(language)
    M = _build_presence_matrix(tok, voc)

    print(f"vocab size: {len(voc)}")
    score_fn = lambda X: _negation_strength(X, N, V)
    configs = [
        ("Singles only (max_size=1)", dict(operator="or", top_k=20, max_size=1)),
        ("OR one expansion round (max_size=2)", dict(operator="or", top_k=20, max_size=2)),
        ("AND one expansion round (max_size=2)", dict(operator="and", top_k=20, max_size=2)),
        ("OR until plateau (max_size=None)", dict(operator="or", top_k=20, max_size=None)),
        ("AND until plateau (max_size=None)", dict(operator="and", top_k=20, max_size=None)),
    ]

    for title, params in configs:
        features = _grow_features(voc, M, score_fn, **params)
        ranked = sorted(((key, score_fn(X)) for key, X in features.items()), key=lambda kv: kv[1], reverse=True)
        max_atoms = max(len(key[1]) for key in features)
        print(f"\n=== {title} ===")
        print(f"features: {len(features)} | max atoms: {max_atoms}")
        print("top 5:")
        for i, (key, score) in enumerate(ranked[:5], start=1):
            op, atoms = key
            print(f"{i:2d}. {op}({','.join(sorted(atoms))}) -> {score:.4f}")
# END DEBUG
