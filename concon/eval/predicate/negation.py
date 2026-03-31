import time
import re
import pandas as pd
import numpy as np

def _tokenize_messages(language: pd.DataFrame):
    assert "msg" in language.columns, "language must have a 'msg' column"
    tokenized = [[t for t in str(msg).split() if t != "0"] for msg in language["msg"]]
    vocab = sorted({t for toks in tokenized for t in toks})
    assert vocab, "vocab cannot be empty"
    return tokenized, vocab

def _build_presence_matrix(tokenized, vocab):
    '''Build binary matrix X of (N_messages, |V|). 
    Set X[i,j]=1 if vocab[j] appears in message i.'''
    X = np.zeros((len(tokenized), len(vocab)), dtype=int)
    tok2idx = {t: i for i, t in enumerate(vocab)}
    for i, toks in enumerate(tokenized):
        for t in toks:
            X[i, tok2idx[t]] = 1
    return X

print("Mock language:")
def _pred_str(neg, idx):
    base = f"P0-v{idx}"
    return f"(¬{base})" if neg else base
rows = []
rows.append({"msg": "1 0", "pred_str": _pred_str(False, 0)})
rows.append({"msg": "17 1 0", "pred_str": _pred_str(True, 0)})
rows.append({"msg": "2 0", "pred_str": _pred_str(False, 1)})
rows.append({"msg": "17 2 0", "pred_str": _pred_str(True, 1)})
rows.append({"msg": "3 4 0", "pred_str": _pred_str(False, 2)})
rows.append({"msg": "17 4 0", "pred_str": _pred_str(True, 2)})
rows.append({"msg": "5 0", "pred_str": _pred_str(False, 3)})
rows.append({"msg": "15 5 0", "pred_str": _pred_str(True, 3)})
# rows.append({"msg": "16 18 0", "pred_str": _pred_str(False, 4)})
# rows.append({"msg": "15 16 0", "pred_str": _pred_str(True, 4)})
toy_set = pd.DataFrame(rows)
print(toy_set)
print()
tok, voc = _tokenize_messages(toy_set)
print(f"Tokenized signals:\n{tok}\nVocabulary:\n{voc}")
X        = _build_presence_matrix(tok, voc)
print(f"Presence matrix for vocabulary:\n{X}")

# Input: language and features
# raw features and the data structure after evaluating disjunctions must be the same
# two cases:
# 1. 
# find such tokens or disjunctions of tokens (== features) that their MI with the predicate values is close to 1, this means they separate the meaning space in half (negation); 
# evaluate each of these disjunctions for 
# precision (number of signals such that the feature is in the signal and the signal designates some negative predicate value / number of negative predicate values), 
# recall (number of signals such that the feature is in the signal and the signal designates some negative predicate value / number of predicate values where the signal occurs), 
# and the XOR criterion (number of predicate values such that [there is a signal that designates the positive value and the feature is in the signal XOR there is a signal that designates the negative value and the feature is in the signal] / total number of predicate values)
# all the above values are in [0,1]
# 2.
# start directly from the XOR criterion, without evaluating MI first; build disjunctions based on the XOR ratio, then we can compute precision and recall
# Let us try and focus on case 2.

def parse_pred(s):
    # Parses predicate labels: negative/positive
    s = str(s)
    m = re.match(r"^\(¬(.+)\)$", s)
    if m:
        return m.group(1), True
    return s, False

# Build features from presence matrix:
# Each token is mapped to its binary feature vector across all messages
# features[("tok", t)] = [False, True, ...]
features = {("tok", t): X[:, j].astype(bool) for j, t in enumerate(voc)}
print(f"Features:\n{features}")

# Prune based on a NAND criterion to keep numbers manageable.
# Indeed, NAND rejects features that appear on both the positive and the negative side for a given predicate value.
# NAND(feature) =
# #(base predicate values v such that
#   [NOT (exists positive signal for v with feature) OR
#    NOT (exists negative signal for v with feature)])
# / #(base predicate values)
vs = []
is_neg = []
for s in toy_set["pred_str"]:
    b, n = parse_pred(s)
    vs.append(b)
    is_neg.append(n)
vs = np.array(vs, dtype=object)
is_neg = np.array(is_neg, dtype=bool)
print(f"whether occurrence is negative:\n{list(zip(vs, is_neg))}")

v_types = sorted(set(vs))
kept = {}

for feature_name, f in features.items():
    # reminder: f is a boolean vector over rows/messages
    nand_hits = 0
    for v in v_types:
        v_mask = (vs == v)
        E_pos_and_feature = np.any(f[v_mask & (~is_neg)])
        E_neg_and_feature = np.any(f[v_mask & is_neg])

        # NAND
        if ((not E_pos_and_feature) or (not E_neg_and_feature)):
            nand_hits += 1

    if nand_hits / len(v_types) == 1:
        kept[feature_name] = f

print(f"NAND kept: {kept}")
# Hereforth only consider pruned elements
features = kept


# Now we evaluate the XOR criterion
# Start with single-token features and score each by XOR
# Keep the features that score high
# Then n times combine retained features with all vocabulary tokens and compute XOR(new_feature), keep only if XOR increased over parent feature
# Augment pool with new features

def _xor_score(f, vs, is_neg, v_types):
    xor_hits = 0
    for v in v_types:
        v_mask = (vs == v)
        E_pos_and_feature = np.any(f[v_mask & (~is_neg)])
        E_neg_and_feature = np.any(f[v_mask & is_neg])
        if E_pos_and_feature ^ E_neg_and_feature:
            xor_hits += 1
    return (xor_hits / len(v_types)) if len(v_types) else 0.0

N = 3
xor_scores, active = {}, {}

for feature_name, f in features.items():
    xor_score = _xor_score(f, vs, is_neg, v_types)
    xor_scores[feature_name] = xor_score
    active[feature_name] = f

feature_pool = dict(features)

for step in range(N):
    print(f"XOR expansion round {step+1}/{N}")
    new_active = {}
    
    for parent_name, parent_f in active.items():
        parent_score = xor_scores[parent_name]

        if parent_name[0] == "tok":
            parent_tokens = {parent_name[1]}
        else:
            parent_tokens = set(parent_name[1])

        for tok_name, tok_f in features.items():
            if tok_name[0] != "tok":
                continue
            t = tok_name[1]
            if t in parent_tokens:
                continue

            child_tokens = tuple(sorted(parent_tokens | {t}))
            child_name = ("or", child_tokens)

            if child_name in feature_pool:
                continue

            child_f = parent_f | tok_f
            child_score = _xor_score(child_f, vs, is_neg, v_types)

            # keep only if not worse than parent (+ above threshold)
            if child_score + 1e-12 >= parent_score:
                feature_pool[child_name] = child_f
                xor_scores[child_name] = child_score
                new_active[child_name] = child_f

    if not new_active:
        print("Nothing new.")
        break
    active = new_active

    print(f"New features:\n{sorted(new_active.keys(), key=lambda n: xor_scores[n], reverse=True)}")

print("\nBest features (10 max):")
for name, s in sorted(xor_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]:
    print(f"{name} -> XOR={s:.3f}")

print(feature_pool)
features = feature_pool

# calculate precision / recall
# also, calculate homogeneity:
# max(
# [number of predicate values s.t. there is a signal that expresses the positive predicate value and the feature belongs to the signal / number of predicate values],
# [number of predicate values s.t. there is a signal that expresses the negative predicate value and the feature belongs to the signal / number of predicate values]
# )
rows = []
n_v = len(v_types)
for name, f in features.items():
    if name[0] == "tok":
        name_str = name[1]
    else:
        name_str = "(" + " ∨ ".join(name[1]) + ")"

    present_in_v, present_in_not_v = [], []
    pos_hits, neg_hits, any_hits = 0, 0, 0

    for v in v_types:
        v_mask = (vs == v)
        E_pos_and_feature = np.any(f[v_mask & (~is_neg)])
        E_neg_and_feature = np.any(f[v_mask & is_neg])

        if E_pos_and_feature:
            present_in_v.append(v)
            pos_hits += 1
            any_hits += 1
        if E_neg_and_feature:
            present_in_not_v.append(f"¬{v}")
            neg_hits += 1
            any_hits += 1
    
    precision   = (neg_hits / any_hits) if any_hits else 0.0
    recall      = (neg_hits / n_v) if n_v else 0.0
    homogeneity = max((pos_hits / n_v) if n_v else 0.0,
                   (neg_hits / n_v) if n_v else 0.0)
    
    rows.append({
        "name": name,
        "name_str": name_str,
        "v": present_in_v,
        "¬v": present_in_not_v,
        "xor_score": xor_scores[name],
        "precision": precision,   # purity
        "recall": recall,         # coverage
        "homogeneity": homogeneity
    })


stats = pd.DataFrame(rows).sort_values(
    ["xor_score", "precision", "recall", "homogeneity"], ascending=False
).reset_index(drop=True)

print(stats[["name_str", "xor_score", "precision", "recall", "homogeneity"]].to_string(index=False))