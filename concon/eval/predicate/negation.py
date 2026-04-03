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

# TODO what's below is generated for ad hoc usability

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def _compute_standalone_negation_stats(language: pd.DataFrame,
                                       expansion_rounds: int = 3,
                                       max_disjunction_size: int = None,
                                       token_top_k: int = None,
                                       show_inner_progress: bool = True,
                                       max_active_per_round: int = None,
                                       max_candidates_total: int = None) -> pd.DataFrame:
    """
    Reproduce the standalone metric computation from this file for one language dataframe.
    """
    assert "pred_str" in language.columns, "language must have a 'pred_str' column"
    tokenized, vocab = _tokenize_messages(language)
    X = _build_presence_matrix(tokenized, vocab)

    features = {("tok", t): X[:, j].astype(bool) for j, t in enumerate(vocab)}

    vs, is_neg = [], []
    for s in language["pred_str"]:
        b, n = parse_pred(s)
        vs.append(b)
        is_neg.append(n)
    vs = np.array(vs, dtype=object)
    is_neg = np.array(is_neg, dtype=bool)
    v_types = sorted(set(vs))

    v_pos_idx = []
    v_neg_idx = []
    for v in v_types:
        v_mask = (vs == v)
        v_pos_idx.append(np.flatnonzero(v_mask & (~is_neg)))
        v_neg_idx.append(np.flatnonzero(v_mask & is_neg))
    # Convert each token feature into predicate-level (pos_present, neg_present) profiles.
    # This avoids rescanning message rows for each candidate disjunction.
    token_profiles = {}
    for feature_name, f in features.items():
        pos_profile = np.array(
            [np.any(f[pos_idx]) if len(pos_idx) else False for pos_idx in v_pos_idx],
            dtype=bool,
        )
        neg_profile = np.array(
            [np.any(f[neg_idx]) if len(neg_idx) else False for neg_idx in v_neg_idx],
            dtype=bool,
        )
        token_profiles[feature_name] = (pos_profile, neg_profile)

    # NAND prune on predicate-level profiles.
    kept = {}
    for feature_name, (pos_profile, neg_profile) in token_profiles.items():
        nand_hits = np.count_nonzero((~pos_profile) | (~neg_profile))
        if nand_hits == len(v_types):
            kept[feature_name] = (pos_profile, neg_profile)

    features = kept

    def _xor_score_fast(pos_profile, neg_profile):
        return float(np.mean(np.logical_xor(pos_profile, neg_profile))) if len(v_types) else 0.0

    def _feature_row(name, pos_profile, neg_profile):
        if name[0] == "tok":
            name_str = name[1]
        else:
            name_str = "(" + " ∨ ".join(name[1]) + ")"
        present_in_v = [v for v, present in zip(v_types, pos_profile) if present]
        present_in_not_v = [f"¬{v}" for v, present in zip(v_types, neg_profile) if present]
        pos_hits = int(np.count_nonzero(pos_profile))
        neg_hits = int(np.count_nonzero(neg_profile))
        any_hits = pos_hits + neg_hits
        n_v = len(v_types)
        precision = (neg_hits / any_hits) if any_hits else 0.0
        recall = (neg_hits / n_v) if n_v else 0.0
        homogeneity = max((pos_hits / n_v) if n_v else 0.0,
                          (neg_hits / n_v) if n_v else 0.0)
        f1_pr = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {
            "name": name,
            "name_str": name_str,
            "v": present_in_v,
            "¬v": present_in_not_v,
            "xor_score": _xor_score_fast(pos_profile, neg_profile),
            "precision": precision,
            "recall": recall,
            "homogeneity": homogeneity,
            "f1_pr": f1_pr,
        }

    # Single-symbol mode: evaluate only token features, no disjunction expansion.
    # Keep compatibility with the function signature by ignoring expansion/search caps.
    all_rows = {
        feature_name: _feature_row(feature_name, pos_profile, neg_profile)
        for feature_name, (pos_profile, neg_profile) in features.items()
        if feature_name[0] == "tok"
    }

    rows = list(all_rows.values())

    if not rows:
        return pd.DataFrame(columns=["name", "name_str", "v", "¬v", "xor_score", "precision", "recall", "homogeneity", "f1_pr"])

    return pd.DataFrame(rows).sort_values(
        ["xor_score", "precision", "recall", "homogeneity"],
        ascending=False
    ).reset_index(drop=True)

def _infer_folder_name(datapoint: dict, run_idx: int, explicit_run_names=None):
    if explicit_run_names is not None and run_idx < len(explicit_run_names):
        v = explicit_run_names[run_idx]
        if v is not None and str(v):
            return str(v)

    for k in ("run_name", "folder_name", "source_folder", "source_dir", "run_dir", "directory"):
        if k in datapoint and datapoint[k]:
            return pathlib.Path(str(datapoint[k])).name

    cfg = datapoint.get("config", {})
    if isinstance(cfg, dict):
        for k in ("run_name", "folder_name", "run_dir", "save_dir", "log_dir"):
            if cfg.get(k):
                return pathlib.Path(str(cfg[k])).name
        run_tag = cfg.get("run_tag")
        run_id = cfg.get("run_id", cfg.get("run"))
        props = cfg.get("properties")
        if run_tag is not None and run_id is not None:
            return f"props={props}__t={run_tag}__run={run_id}" if props is not None else f"{run_tag}__run={run_id}"
        if run_tag is not None:
            return str(run_tag)

    return f"run_{run_idx}"

def _compact_run_label(run_name: str, run_idx: int) -> str:
    run_name = str(run_name)
    m_props = re.search(r"props=(\d+)", run_name)
    m_run = re.search(r"__run=(\d+)", run_name)
    m_tag = re.search(r"__t=([0-9]{4}-[0-9]{2}-[0-9]{2})", run_name)
    if m_props and m_run:
        tag = f"-{m_tag.group(1)}" if m_tag else ""
        return f"p{m_props.group(1)}-r{m_run.group(1)}{tag}"
    if len(run_name) > 24:
        return run_name[:24]
    return run_name or f"run{run_idx}"

def _search_limits_from_vocab(vocab_size: int, profile: str = "fast"):
    """
    Adaptive bounds for search complexity by vocabulary size and profile.
    Returns (max_disjunction_size, token_top_k, max_active_per_round, max_candidates_total).
    """
    v = int(vocab_size)
    p = str(profile).strip().lower()
    if p == "slow":
        # More permissive search, still bounded by structural caps.
        if v >= 2000:
            return 6, 48, 7000, 140000
        if v >= 800:
            return 7, 64, 10000, 220000
        if v >= 300:
            return 8, 80, 14000, 320000
        if v >= 120:
            return 10, 96, 20000, 450000
        return 11, 120, 26000, 600000

    # Fast baseline profile.
    if v >= 1200:
        return 2, 16, 1200, 12000
    if v >= 400:
        return 3, 16, 1600, 16000
    if v >= 150:
        return 4, 20, 2000, 22000
    return 5, 24, 2600, 28000

def _analyze_single_datapoint_for_export(run_idx: int,
                                         datapoint: dict,
                                         run_names,
                                         expansion_rounds: int,
                                         top_k: int,
                                         show_inner_progress: bool,
                                         bin_profile: str = "single_symbol"):
    cfg = datapoint.get("config", {}) if isinstance(datapoint, dict) else {}
    folder_name = _infer_folder_name(datapoint, run_idx, explicit_run_names=run_names)
    base_row = {
        "run_idx": run_idx,
        "run_name": folder_name,
        "properties": cfg.get("properties"),
        "hidden_size": cfg.get("hidden_size"),
        "num_predicates_cfg": cfg.get("num_predicates"),
        "no_negation": cfg.get("no_negation"),
        "no_conjunction": cfg.get("no_conjunction"),
    }

    languages = datapoint.get("languages") if isinstance(datapoint, dict) else None
    if not languages:
        return [], {**base_row, "status": "missing_language"}

    latest_lang = max(languages, key=lambda x: x.get("epoch_number", -1))
    epoch_number = latest_lang.get("epoch_number", -1)
    language = latest_lang.get("language")
    if language is None or len(language) == 0:
        return [], {**base_row, "epoch": epoch_number, "status": "empty_language"}
    if "msg" not in language.columns or "pred_str" not in language.columns:
        return [], {**base_row, "epoch": epoch_number, "status": "missing_msg_or_pred_str"}

    vocab_size = len({t for msg in language["msg"] for t in str(msg).split() if t != "0"})
    try:
        stats = _compute_standalone_negation_stats(
            language,
            expansion_rounds=0,
            max_disjunction_size=1,
            token_top_k=None,
            show_inner_progress=show_inner_progress,
            max_active_per_round=None,
            max_candidates_total=None,
        )
    except Exception as e:
        return [], {**base_row, "epoch": epoch_number, "status": f"error: {e}"}

    n_messages = len(language)
    n_predicates = len({parse_pred(s)[0] for s in language["pred_str"]})
    base_row = {
        **base_row,
        "epoch": epoch_number,
        "n_messages": n_messages,
        "n_predicates": n_predicates,
        "vocab_size": vocab_size,
        "candidate_mode": "single_symbol",
        "max_disjunction_size": 1,
        "token_top_k": None,
        "max_active_per_round": None,
        "max_candidates_total": None,
        "bin_profile": "single_symbol",
    }

    if stats.empty:
        return [], {**base_row, "status": "no_candidates"}

    analysis_rows = []
    top_stats = stats.head(top_k).reset_index(drop=True)
    for rank, row in enumerate(top_stats.to_dict(orient="records"), start=1):
            analysis_rows.append({
                **base_row,
                "rank": rank,
                "name_str": row["name_str"],
                "xor_score": row["xor_score"],
                "precision": row["precision"],
                "recall": row["recall"],
                "homogeneity": row["homogeneity"],
                "f1_pr": row["f1_pr"],
                "v": row["v"],
                "¬v": row["¬v"],
            })

    best = top_stats.iloc[0].to_dict()
    summary_row = {
        **base_row,
        "status": "ok",
        "n_candidates_total": len(stats),
        "name_str": best["name_str"],
        "xor_score": best["xor_score"],
        "precision": best["precision"],
        "recall": best["recall"],
        "homogeneity": best["homogeneity"],
        "f1_pr": best["f1_pr"],
        "v": best["v"],
        "¬v": best["¬v"],
    }
    return analysis_rows, summary_row

def export_negation_metrics_csvs(datapoints_list,
                                 *,
                                 top_k: int = 10,
                                 out_dir: str = None,
                                 experiment_name: str = "experiment",
                                 run_names=None,
                                 expansion_rounds: int = 3,
                                 n_jobs: int = 1,
                                 bin_profile: str = "single_symbol"):
    """
    Compute standalone negation metrics for the latest language in each datapoint and export:
    1) analysis CSV with top-k items per datapoint,
    2) summary CSV with only the top item per datapoint.
    Candidates are restricted to single vocabulary symbols (no disjunction expansion).
    Returns (analysis_df, summary_df, analysis_path, summary_path).
    """
    top_k = max(1, int(top_k))
    n_jobs = min(4, max(1, int(n_jobs)))
    out_dir_path = pathlib.Path(out_dir) if out_dir is not None else pathlib.Path(__file__).resolve().parent / "outputs"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    analysis_rows = []
    summary_rows = []

    datapoints_seq = list(datapoints_list)
    total_runs = len(datapoints_seq)

    if n_jobs == 1:
        run_iter = tqdm(
            enumerate(datapoints_seq),
            total=total_runs,
            desc="Negation analysis",
        )
        for run_idx, datapoint in run_iter:
            run_iter.set_postfix_str(_infer_folder_name(datapoint, run_idx, explicit_run_names=run_names))
            run_analysis_rows, run_summary = _analyze_single_datapoint_for_export(
                run_idx,
                datapoint,
                run_names=run_names,
                expansion_rounds=expansion_rounds,
                top_k=top_k,
                show_inner_progress=True,
                bin_profile=bin_profile,
            )
            analysis_rows.extend(run_analysis_rows)
            if run_summary is not None:
                summary_rows.append(run_summary)
    else:
        progress = tqdm(total=total_runs, desc=f"Negation analysis ({n_jobs} jobs)")
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _analyze_single_datapoint_for_export,
                    run_idx,
                    datapoint,
                    run_names,
                    expansion_rounds,
                    top_k,
                    False,
                    bin_profile,
                ): run_idx
                for run_idx, datapoint in enumerate(datapoints_seq)
            }
            for future in as_completed(futures):
                run_analysis_rows, run_summary = future.result()
                analysis_rows.extend(run_analysis_rows)
                if run_summary is not None:
                    summary_rows.append(run_summary)
                    short_name = _compact_run_label(run_summary.get("run_name", ""), int(run_summary.get("run_idx", -1)))
                    progress.set_postfix_str(short_name)
                progress.update(1)
        progress.close()

    analysis_df = pd.DataFrame(analysis_rows).sort_values(
        ["run_idx", "rank"], ascending=[True, True]
    ).reset_index(drop=True) if analysis_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["run_idx"], ascending=[True]
    ).reset_index(drop=True) if summary_rows else pd.DataFrame()

    analysis_path = out_dir_path / f"negation_analysis_topk_{experiment_name}.csv"
    summary_path = out_dir_path / f"negation_summary_top1_{experiment_name}.csv"
    analysis_df.to_csv(analysis_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved analysis CSV: {analysis_path}")
    print(f"Saved summary CSV:  {summary_path}")
    return analysis_df, summary_df, analysis_path, summary_path
