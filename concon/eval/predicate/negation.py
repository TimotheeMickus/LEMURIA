import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import re
import sys
from load import get_datapoints
import os
from tqdm import tqdm
import multiprocessing as mp

TQDM_ENABLED = sys.stderr.isatty() # reasonable outputs in IDE

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

def _evaluate_per_token_disjunctions(current_item, vocab_features, y, max_size, results):
    '''Recursively explores disjunctions containing a fixed element of the vocabulary.'''
    if current_item['size'] >= max_size:
        return

    for tok, feat in tqdm(vocab_features.items(), total=len(vocab_features), desc="MI disjunctions", disable=not TQDM_ENABLED, leave=False):
        if tok in current_item['tokens']: # Avoid adding tokens already in the disjunction set
            continue

        new_feat = (current_item['feat'].astype(bool) | feat.astype(bool)).astype(int) # Binary feature vector for the disjunction
        new_mi = mutual_info_score(new_feat, y) / np.log(2) # Calculate the MI between the feature vector and predicate labels

        # If MI improved, recurse deeper from the newly found disjunction
        if new_mi > current_item['mi_bits']:
            new_tokens = current_item['tokens'] | {tok}
            new_item = {
                'name': f"({' ∨ '.join(sorted(new_tokens))})",
                'tokens': new_tokens,
                'feat': new_feat,
                'mi_bits': new_mi,
                'size': len(new_tokens)
            }
            results.append(new_item)
            _evaluate_per_token_disjunctions(new_item, vocab_features, y, max_size, results)

def _filter_candidates(
    merged: pd.DataFrame,
    *,
    min_purity: float,
    min_coverage: float,
    min_exclusive_rate: float = 1.0,
):
    '''Filter candidates by purity/coverage/exclusivity thresholds.
    This helper is used in both greedy and full searches.'''
    if merged.empty:
        return merged, {"status": "no_candidates", "max_purity": None, "max_coverage": None}
    max_purity = float(merged["neg_purity"].max())
    max_coverage = float(merged["neg_coverage"].max())
    max_exclusive = float(merged["exclusive_rate"].max())
    merged = merged[
        (merged["neg_purity"] >= min_purity)
        & (merged["neg_coverage"] >= min_coverage)
        & (merged["exclusive_rate"] >= min_exclusive_rate)
    ]
    if merged.empty:
        return merged, {"status": "below_threshold", "max_purity": max_purity, "max_coverage": max_coverage, "max_exclusive_rate": max_exclusive}
    return merged, {"status": "pass", "max_purity": max_purity, "max_coverage": max_coverage, "max_exclusive_rate": max_exclusive}

def _coverage_stats(language: pd.DataFrame, disjunctions: pd.DataFrame):
    '''Compute purity, coverage, and exclusivity rate for each feature.'''
    if("name" not in disjunctions.columns): raise KeyError("disjunctions missing column 'name'")

    # Map (pred: {v: [tok], ¬v: [tok]}).
    base_map = {}
    for _, row in language.iterrows():
        pred_str = str(row["pred_str"])
        msg = str(row["msg"])
        tokens = [t for t in msg.split() if t != "0"]
        is_neg = "¬" in pred_str
        base = pred_str.replace("(", "").replace(")", "").replace("¬", "")
        entry = base_map.setdefault(base, {"pos": [], "neg": []})
        entry["neg" if is_neg else "pos"].append(tokens)

    rows = []
    for item in disjunctions["name"]:
        tok_set = set(re.findall(r"[A-Za-z0-9_]+", str(item))) # parse disjunction string ("(a ∨ b)" --> {"a","b"})
        present_in_v = []
        present_in_not_v = []
        both_count = 0

        for base, entry in base_map.items():
            # Check whether the disjunction appears in v-messages or ¬v-messages.
            present_v = any(any(t in tok_set for t in toks) for toks in entry["pos"])
            present_not_v = any(any(t in tok_set for t in toks) for toks in entry["neg"])
            if(present_v): present_in_v.append(base)
            if(present_not_v): present_in_not_v.append(f"¬{base}")
            if(present_v and present_not_v): both_count += 1

        # any_present counts bases where the disjunction appears in either v or ¬v.
        v_only_count = len(present_in_v) - both_count
        not_v_only_count = len(present_in_not_v) - both_count
        any_present = v_only_count + not_v_only_count + both_count
        # neg_purity: among bases where it appears, how often is it only ¬v?
        neg_purity = (not_v_only_count / any_present) if any_present else 0.0
        # neg_coverage: fraction of all bases where it marks ¬v.
        neg_coverage = (not_v_only_count / len(base_map)) if len(base_map) else 0.0
        # exclusive_rate: fraction of predicates where the feature does not appear on both sides.
        exclusive_rate = (1.0 - (both_count / any_present)) if any_present else 0.0

        rows.append({
            "name": item,
            "v": present_in_v,
            "¬v": present_in_not_v,
            "neg_purity": neg_purity,
            "neg_coverage": neg_coverage,
            "exclusive_rate": exclusive_rate,
        })

    return pd.DataFrame(rows)

def _process_run(args):
    # Avoid noisy nested progress bars in worker processes.
    global TQDM_ENABLED
    TQDM_ENABLED = False
    run_idx, datapoint, best_language, mode, max_size, min_purity, min_coverage, min_exclusive_rate, max_vocab_for_full, greedy_top_k = args
    config = datapoint.get("config", {})
    language_df = best_language["language"]
    _, vocab = _tokenize_messages(language_df)
    vocab_size = len(vocab)

    # Build candidate disjunctions using one search strategy.
    disjunctions = None
    if mode == "greedy":
        disjunctions = mi_maximizing_disjunctions_greedy(language_df, max_size=max_size, top_k=greedy_top_k)
    else:
        if vocab_size <= max_vocab_for_full:
            disjunctions = mi_maximizing_disjunctions(language_df, max_size=max_size)
        else:
            disjunctions = None

    if disjunctions is None or disjunctions.empty:
        passed = pd.DataFrame()
        stats = {
            "status": "skipped_vocab" if mode == "full" and vocab_size > max_vocab_for_full else "no_disjunctions",
            "max_purity": None,
            "max_coverage": None,
        }
        score_mi, score_xor, score_purity = None, None, None
    else:
        score_mi = float(disjunctions["mi_bits"].max())
        merged = disjunctions.merge(_coverage_stats(language_df, disjunctions), on="name", how="inner")
        score_xor = float(merged["exclusive_rate"].max()) if not merged.empty else None
        score_purity = float(merged["neg_purity"].max()) if not merged.empty else None
        passed, stats = _filter_candidates(
            merged,
            min_purity=min_purity,
            min_coverage=min_coverage,
            min_exclusive_rate=min_exclusive_rate,
        )

    # Keep all passing candidates and note the best one for comparison.
    candidates = ";".join(passed["name"].astype(str).tolist()) if not passed.empty else ""
    best_candidate, best_purity, best_coverage = None, None, None
    if not passed.empty:
        best_row = passed.sort_values(["neg_purity", "neg_coverage"], ascending=False).iloc[0]
        best_candidate = best_row["name"]
        best_purity = float(best_row["neg_purity"])
        best_coverage = float(best_row["neg_coverage"])

    return {
        "run_idx": run_idx,
        "run_name": f"{config.get('properties')}_{config.get('run_tag')}",
        "epoch": int(best_language.get("epoch_number", -1)),
        "complexity": config.get("num_predicates"),
        "properties": config.get("properties"),
        "hidden_size": config.get("hidden_size"),
        "vocab_size": vocab_size,
        "n_messages": int(len(language_df)),
        "n_predicates": int(language_df["pred_str"].nunique()),
        "n_neg_predicates": int(language_df["pred_str"].astype(str).str.contains("¬").sum()),

        "status": stats.get("status"),
        "max_purity": stats.get("max_purity"),
        "max_coverage": stats.get("max_coverage"),
        "max_exclusive_rate": stats.get("max_exclusive_rate"),
        "n_pass": int(len(passed)),
        "candidates": candidates,
        "best_candidate": best_candidate,
        "best_purity": best_purity,
        "best_coverage": best_coverage,
        "score_mi": score_mi,
        "score_xor": score_xor,
        "score_purity": score_purity,
    }

def mi_maximizing_disjunctions(language, max_size=None):
    '''For all elements in the language's vocabulary, recursively build disjunctions up to max_size elements and record I(disj ; predicates).'''
    tokenized, vocab = _tokenize_messages(language)
    X = _build_presence_matrix(tokenized, vocab)
    y = language['pred_str'].to_numpy() # predicate label per signal

    # Set max_size to length of vocabulary unless specified.
    if(max_size is None): max_size = len(vocab)
    # Each element in the vocabulary is paired with a binary feature vector of whether it expresses some predicate.
    vocab_features = {vocab[j]: X[:,j].astype(int) for j in range(len(vocab))}

    results = []
    for tok, feat in vocab_features.items():
        # MI between "token present?" and predicate identity.
        mi = mutual_info_score(feat, y) / np.log(2)
        # Build a base item for each single element of the vocabulary.
        base_item = {
            'name': tok,
            'tokens': {tok},
            'feat': feat,
            'mi_bits': mi,
            'size': 1
        }
        results.append(base_item)
        # Recursively build disjunctions starting from the base item if they improve MI(disj ; pred. ident.)
        _evaluate_per_token_disjunctions(base_item, vocab_features, y, max_size, results)

    return pd.DataFrame(results).drop(columns=['feat', 'tokens']).sort_values('mi_bits', ascending=False).drop_duplicates('name')

def mi_maximizing_disjunctions_greedy(language, max_size=None, top_k=1):
    '''Start from the top-k MI tokens, then iteratively add the largest-MI gain token.'''
    tokenized, vocab = _tokenize_messages(language)
    X = _build_presence_matrix(tokenized, vocab)
    y = language['pred_str'].to_numpy()

    if(max_size is None): max_size = len(vocab)

    mi_per_tok = []
    for j in tqdm(range(len(vocab)), desc="MI greedy seed", disable=not TQDM_ENABLED, leave=False):
        mi = mutual_info_score(X[:, j], y) / np.log(2)
        mi_per_tok.append(mi)

    if(not mi_per_tok): return pd.DataFrame(columns=["name", "mi_bits", "size"])

    top_k = max(1, min(int(top_k), len(vocab)))
    seeds = np.argsort(mi_per_tok)[::-1][:top_k]

    rows = []
    for seed in seeds:
        best_mi = mi_per_tok[seed]
        selected = [seed]
        rows.append({"name": vocab[seed], "mi_bits": best_mi, "size": 1})

        # Record which signals are covered by the current disjunction.
        remaining = [j for j in range(len(vocab)) if j not in selected]
        current_feat = X[:, seed].astype(bool)

        # Iteratively add the token with the largest MI gain.
        while remaining and len(selected) < max_size:
            best_gain, best_ids, best_feat = None, None, None
            for j in tqdm(remaining, desc="MI greedy expand", disable=not TQDM_ENABLED, leave=False):
                disj = (current_feat | X[:, j].astype(bool)) # candidate disjunction is (current set | token j)
                mi = mutual_info_score(disj.astype(int), y) / np.log(2)
                gain = mi - best_mi
                if best_gain is None or gain > best_gain:
                    best_gain, best_idx, best_feat = gain, j, disj

            if(best_gain is None): break

            selected.append(best_idx)
            remaining.remove(best_idx)
            best_mi = best_mi + best_gain
            current_feat = best_feat

            # Log the best disjunction at this size.
            rows.append({
                "name": f"({' ∨ '.join(sorted(vocab[i] for i in selected))})",
                "mi_bits": best_mi,
                "size": len(selected),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("mi_bits", ascending=False)
        .drop_duplicates("name")
        .reset_index(drop=True)
    )

def xor_coverage(language: pd.DataFrame, disjunctions: pd.DataFrame):
    '''For each feature (disjunction) and presence of negation v, compute if it verifies v ⊕ ¬v, and for how many predicates.'''
    stats_df = _coverage_stats(language, disjunctions)
    # Keep only disjunctions that mark exactly v ⊕ ¬v.
    has_xor = stats_df["v"].str.len().gt(0) ^ stats_df["¬v"].str.len().gt(0)
    stats_df = stats_df[has_xor]
    return disjunctions.merge(stats_df, on="name", how="inner")

def find_negation(language: pd.DataFrame, max_size=None, *, mode: str = 'greedy', min_purity: float = 1.0, min_coverage: float = 1.0, min_exclusive_rate: float = 1.0, greedy_top_k: int = 1):
    '''In a language, find a feature that behaves like a negation marker. Returns single best candidate.'''
    assert mode in ['greedy', 'full'], "Choose either \"greedy\" or \"full\" mode."

    if mode == 'greedy':
        disjunctions = mi_maximizing_disjunctions_greedy(language, max_size=max_size, top_k=greedy_top_k)
    elif mode == 'full':
        disjunctions = mi_maximizing_disjunctions(language, max_size=max_size)

    merged = disjunctions.merge(_coverage_stats(language, disjunctions), on="name", how="inner")
    if merged.empty:
        print("No candidate features found.\n")
        return None
    
    passed, stats = _filter_candidates(
        merged,
        min_purity=min_purity,
        min_coverage=min_coverage,
        min_exclusive_rate=min_exclusive_rate,
    )
    if passed.empty:
        print(f"No candidates pass thresholds. max purity={stats['max_purity']:.3f}, max coverage={stats['max_coverage']:.3f}\n")
        return None
    
    best = passed.sort_values(["neg_purity", "neg_coverage"], ascending=False).iloc[0]
    return best["name"], best["¬v"]


def compare_greedy_exhaustive_negation_search(datapoints, *, max_size=4, min_purity=1.0, min_coverage=1.0, min_exclusive_rate=1.0, max_vocab_for_full=12, n_jobs=4, greedy_top_k: int = 1):
    '''Applies negation search using greedy and exhaustive approaches and compares them.'''
    # Check if there are any eligible runs at all
    if all(dp.get("config", {}).get("no_negation", False) or not dp.get("languages") for dp in datapoints):
        print("No eligible runs for negation analysis; skipping.")
        return pd.DataFrame()

    greedy_df = run_negation_search(
        datapoints,
        mode="greedy",
        max_size=max_size,
        min_purity=min_purity,
        min_coverage=min_coverage,
        min_exclusive_rate=min_exclusive_rate,
        max_vocab_for_full=max_vocab_for_full,
        n_jobs=n_jobs,
        greedy_top_k=greedy_top_k,
    )
    full_df = run_negation_search(
        datapoints,
        mode="full",
        max_size=max_size,
        min_purity=min_purity,
        min_coverage=min_coverage,
        min_exclusive_rate=min_exclusive_rate,
        max_vocab_for_full=max_vocab_for_full,
        n_jobs=n_jobs,
    )

    key_cols = ["run_idx", "run_name", "epoch", "complexity", "properties", "hidden_size", "vocab_size", "n_messages", "n_predicates", "n_neg_predicates"]
    metric_cols = [
        "status", "max_purity", "max_coverage", "max_exclusive_rate",
        "n_pass", "candidates",
        "best_candidate", "best_purity", "best_coverage",
        "score_mi", "score_xor", "score_purity",
    ]

    greedy_df = greedy_df.rename(columns={c: f"greedy_{c}" for c in metric_cols})
    full_df = full_df.rename(columns={c: f"full_{c}" for c in metric_cols})

    merged = pd.merge(greedy_df, full_df, on=key_cols, how="outer")
    merged["same_best"] = (
        merged["greedy_best_candidate"].notna()
        & merged["full_best_candidate"].notna()
        & (merged["greedy_best_candidate"] == merged["full_best_candidate"])
    )

    return merged


def run_negation_search(datapoints, *, mode="greedy", max_size=4, min_purity=1.0, min_coverage=1.0, min_exclusive_rate=1.0, max_vocab_for_full=12, n_jobs=4, greedy_top_k: int = 1):
    '''
    Run a single negation-search mode across all datapoints and return a per-run summary.
    - mode="greedy": heuristic search over disjunctions
    - mode="full": exhaustive (until vocab_size > max_vocab_for_full) disjunction search
    Filters candidates by purity/coverage/exclusivity thresholds and records all passing candidates plus the best one.
    Returns DataFrame.
    '''
    assert mode in ["greedy", "full"], "mode must be 'greedy' or 'full'"
    eligible_runs = []
    skipped = {"no_negation": 0}

    for datapoint in datapoints:
        config = datapoint.get("config", {})
        if(config.get("no_negation", False)): skipped["no_negation"] += 1; continue
        languages = datapoint.get("languages") or []
        best_language = max(languages, key=lambda x: x.get("epoch_number", -1))
        eligible_runs.append((datapoint, best_language))

    print(f"Negation search ({mode}): {len(eligible_runs)}/{len(datapoints)} eligible "
          f"(no_negation={skipped['no_negation']})\n")

    args = [
        (run_idx, datapoint, best_language, mode, max_size, min_purity, min_coverage, min_exclusive_rate, max_vocab_for_full, greedy_top_k)
        for run_idx, (datapoint, best_language) in enumerate(eligible_runs)
    ]
    if n_jobs and n_jobs > 1:
        with mp.Pool(processes=n_jobs) as pool:
            rows = list(tqdm(pool.imap(_process_run, args), total=len(args), desc=f"Negation search ({mode}, {n_jobs} jobs)"))
    else:
        rows = [
            _process_run(arg)
            for arg in tqdm(args, desc=f"Negation search ({mode})")
        ]

    return pd.DataFrame(rows)

# def symbol_predicate_mi(language: pd.DataFrame):
#     '''Compute mutual information between symbols and predicate identities.'''
#     tokenized, vocab = _tokenize_messages(language)
#     X = _build_presence_matrix(tokenized, vocab)
#     y = language["pred_str"].to_numpy()

#     # MI per single token (bits)
#     mi = [{"token": tok, "mi_bits": mutual_info_score(y, X[:, j]) / np.log(2)} for j, tok in enumerate(vocab)]
#     return pd.DataFrame(mi).sort_values("mi_bits", ascending=False).reset_index(drop=True)

def normalize_negation_df(neg_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Ensure negation analysis frames have mode-prefixed columns for plotting.
    If prefixed columns already exist, returns a copy unchanged; otherwise
    adds prefixed aliases for base columns when present.
    """
    prefix = f"{mode}_"
    base_cols = [
        "status", "max_purity", "max_coverage", "max_exclusive_rate", "n_pass", "candidates",
        "best_candidate", "best_purity", "best_coverage", "score_mi", "score_xor", "score_purity"
        ]
    df = neg_df.copy()
    # If at least one prefixed col exists, only fill missing prefixed cols.
    for base in base_cols:
        pref = f"{prefix}{base}"
        if pref not in df.columns and base in df.columns:
            df[pref] = df[base]
    return df


if __name__ == "__main__":
    path = input("Input language path or \"demo\" for demonstration: ")
    if path == "demo":
        import time
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
        print("Greedy MI disjunctions (timed, max_size=4):")
        t0 = time.time()
        greedy_mi = mi_maximizing_disjunctions_greedy(toy_set, max_size=4)
        t1 = time.time()
        print(greedy_mi)
        print(f"Greedy MI time: {t1 - t0:.4f}s")
        print()

        print("Exhaustive MI disjunctions (timed, max_size=4):")
        t0 = time.time()
        exhaustive_mi = mi_maximizing_disjunctions(toy_set, max_size=4)
        t1 = time.time()
        print(exhaustive_mi.head(10))
        print(f"Exhaustive MI time: {t1 - t0:.4f}s")
        print()

        print("XOR/coverage stats for greedy MI disjunctions:")
        mi_greedy_stats = xor_coverage(toy_set, greedy_mi)
        print(mi_greedy_stats.head(10))
        print()

        print("XOR/coverage stats for exhaustive MI disjunctions:")
        mi_exhaustive_stats = xor_coverage(toy_set, exhaustive_mi)
        print(mi_exhaustive_stats.sort_values("neg_purity", ascending=False).head(10))
        print()

        def _find_negation(df, name_col, purity_thr=1.0, coverage_thr=1.0):
            mask = (df["neg_purity"] >= purity_thr) & (df["neg_coverage"] >= coverage_thr)
            if not mask.any():
                return None
            best = df[mask].sort_values(["neg_purity", "neg_coverage"], ascending=False).iloc[0]
            return best[name_col], best["¬v"]

        purity_thr = 1.0
        coverage_thr = 1.0

        greedy_negation = _find_negation(mi_greedy_stats, "name", purity_thr, coverage_thr)
        exhaustive_negation = _find_negation(mi_exhaustive_stats, "name", purity_thr, coverage_thr)

        print("Greedy negation:", greedy_negation)
        print("Exhaustive negation:", exhaustive_negation)

    else:
        def _run_find_negation(lang, label, vocab_used=None):
            #DEBUG 
            print(f"[DEBUG] Running negation check for: {label}")
            print(f"\n=== {label} | rows={len(lang)} ===")
            neg_greedy = find_negation(lang, max_size=4, mode="greedy", min_purity=1.0, min_coverage=1.0)
            #DEBUG 
            print(f"[DEBUG] Greedy negation result for {label}: {neg_greedy}")
            print("Greedy negation:", neg_greedy)
            if vocab_used is None:
                _, vocab = _tokenize_messages(lang)
                vocab_used = len(vocab)
            #DEBUG 
            print(f"[DEBUG] vocab_used={vocab_used}")
            if vocab_used <= 12:
                #DEBUG 
                print(f"[DEBUG] Vocab size {vocab_used} <= 12; running exhaustive.")
                neg_full = find_negation(lang, max_size=4, mode="full", min_purity=1.0, min_coverage=1.0)
                print("Exhaustive negation:", neg_full)
            else:
                #DEBUG 
                print(f"[DEBUG] Vocab size {vocab_used} > 12; skipping exhaustive.")
                print(f"Skipping exhaustive negation (vocab size={vocab_used}).")

        # If a CSV path is provided, load that single language file.
        if path.endswith(".csv"):
            lang = pd.read_csv(path)
            print(f"Loaded language: {path}")
            _run_find_negation(lang, os.path.basename(path))
        else:
            # Otherwise treat input as an experiment folder inside runs/.
            datapoints = get_datapoints(path)
            if not datapoints:
                raise FileNotFoundError(f"No datapoints found for experiment '{path}'.")

            # Use the latest language snapshot for each run.
            for idx, d in enumerate(datapoints):
                if not d["languages"]:
                    #DEBUG 
                    print(f"[DEBUG] Run {idx}: no languages, skipping.")
                    continue
                cfg = d.get("config", {})
                if cfg.get("no_negation", False):
                    #DEBUG 
                    print(f"[DEBUG] Run {idx}: negation disabled, skipping.")
                    continue
                latest = max(d["languages"], key=lambda x: x["epoch_number"])
                #DEBUG 
                print(f"[DEBUG] Run {idx}: using epoch {latest['epoch_number']}.")
                eval_df = d.get("evaluation")
                vocab_used = None
                if eval_df is not None and "eval/vocab_used" in eval_df.columns:
                    vocab_used = int(eval_df["eval/vocab_used"].iloc[-1])
                _run_find_negation(
                    latest["language"],
                    f"Run {idx} | epoch {latest['epoch_number']}",
                    vocab_used=vocab_used
                )
