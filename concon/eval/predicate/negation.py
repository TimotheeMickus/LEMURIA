import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import re
import sys
from load import get_datapoints
import os
from tqdm import tqdm

TQDM_ENABLED = sys.stderr.isatty() # reasonable outputs in IDE

#
# ------ Private functions ------ 
#

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

def _filter_xor_candidates(disjunctions: pd.DataFrame, language: pd.DataFrame, *, min_purity: float, min_coverage: float):
    '''Take `xor_coverage` output and pass/fail candidates based on purity and coverage.
    This helper is used in both greedy and full searches.'''
    df = xor_coverage(language, disjunctions)
    if(df.empty): return df, {"status": "no_xor_candidates", "max_purity": None, "max_coverage": None}
    max_purity, max_coverage = float(df["neg_purity"].max()), float(df["neg_coverage"].max())
    df = df[(df["neg_purity"] >= min_purity) & (df["neg_coverage"] >= min_coverage)] # Apply purity/coverage mask to dataframe
    if(df.empty): return df, {"status": "below_threshold", "max_purity": max_purity, "max_coverage": max_coverage}
    return df, {"status": "pass", "max_purity": max_purity, "max_coverage": max_coverage}


#
# ------ Main functions. ------
#

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

def mi_maximizing_disjunctions_greedy(language, max_size=None):
    '''Start from the highest-MI token, then iteratively add the largest-MI gain token.'''
    tokenized, vocab = _tokenize_messages(language)
    X = _build_presence_matrix(tokenized, vocab)
    y = language['pred_str'].to_numpy()

    if(max_size is None): max_size = len(vocab)

    # Pick the token with the highest MI.
    best_j = None
    best_mi = None
    for j in tqdm(range(len(vocab)), desc="MI greedy seed", disable=not TQDM_ENABLED, leave=False):
        mi = mutual_info_score(X[:, j], y) / np.log(2)
        if best_mi is None or mi > best_mi:
            best_mi = mi
            best_j = j

    if(best_j is None): return pd.DataFrame(columns=["name", "mi_bits", "size"])

    selected = [best_j]
    rows = [{"name": vocab[best_j], "mi_bits": best_mi, "size": 1}]

    # Record which signals are covered by the current disjunction.
    remaining = [j for j in range(len(vocab)) if j not in selected]
    current_feat = X[:, best_j].astype(bool)

    # Iteratively add the token with the largest MI gain.
    while remaining and len(selected) < max_size:
        best_gain = None
        best_idx = None
        best_feat = None
        for j in tqdm(remaining, desc="MI greedy expand", disable=not TQDM_ENABLED, leave=False):
            disj = (current_feat | X[:, j].astype(bool)) # candidate disjunction is (current set | token j)
            mi = mutual_info_score(disj.astype(int), y) / np.log(2)
            gain = mi - best_mi
            if best_gain is None or gain > best_gain:
                best_gain, best_idx, best_feat = gain, j, disj

        if(best_gain is None): break # Halt if no improvement.

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

    return pd.DataFrame(rows).sort_values('mi_bits', ascending=False).reset_index(drop=True)

def xor_coverage(language: pd.DataFrame, disjunctions: pd.DataFrame):
    '''For each feature (disjunction) and presence of negation v, compute if it verifies v ⊕ ¬v, and for how many predicates.'''
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
            if present_v:
                present_in_v.append(base)
            if present_not_v:
                present_in_not_v.append(f"¬{base}")
            if present_v and present_not_v:
                both_count += 1

        # any_present counts bases where the disjunction appears in either v or ¬v.
        v_only_count = len(present_in_v) - both_count
        not_v_only_count = len(present_in_not_v) - both_count
        any_present = v_only_count + not_v_only_count + both_count
        # neg_purity: among bases where it appears, how often is it only ¬v?
        neg_purity = (not_v_only_count / any_present) if any_present else 0.0
        # neg_coverage: fraction of all bases where it marks ¬v.
        neg_coverage = (not_v_only_count / len(base_map)) if len(base_map) else 0.0

        rows.append({
            "name": item,
            "v": present_in_v,
            "¬v": present_in_not_v,
            "neg_purity": neg_purity,
            "neg_coverage": neg_coverage,
        })

    stats_df = pd.DataFrame(rows)
    # Keep only disjunctions that mark exactly v ⊕ ¬v.
    has_xor = stats_df["v"].str.len().gt(0) ^ stats_df["¬v"].str.len().gt(0)
    stats_df = stats_df[has_xor]
    return disjunctions.merge(stats_df, on="name", how="inner")

def find_negation(language: pd.DataFrame, max_size=None, *, mode: str = 'greedy', min_purity: float = 1.0, min_coverage: float = 1.0):
    '''In a language, find a feature that behaves like a negation marker. Returns single best candidate.'''
    assert mode in ['greedy', 'full'], "Choose either \"greedy\" or \"full\" mode."

    if mode == 'greedy':
        greedy_disjunctions = mi_maximizing_disjunctions_greedy(language, max_size=max_size)
        df = xor_coverage(language, greedy_disjunctions)
    elif mode == 'full':
        all_disjunctions = mi_maximizing_disjunctions(language, max_size=max_size)
        df = xor_coverage(language, all_disjunctions)

    if df.empty:
        print("No candidate feature fits the XOR criteria.\n")
        return None
    
    mask = (df["neg_purity"] >= min_purity) & (df["neg_coverage"] >= min_coverage)
    if not mask.any():
        print(f"No candidates pass thresholds. max purity={df['neg_purity'].max():.3f}, max coverage={df['neg_coverage'].max():.3f}\n")
        return None
    
    best = df[mask].sort_values(["neg_purity", "neg_coverage"], ascending=False).iloc[0]
    return best["name"], best["¬v"]


def compare_greedy_exhaustive_negation_search(datapoints, *, max_size=4, min_purity=1.0, min_coverage=1.0, max_vocab_for_full=12):
    '''Applies negation search using greedy and exhaustive approaches and compares them.'''
    greedy_df, skipped, n_eligible = run_negation_search(
        datapoints,
        mode="greedy",
        max_size=max_size,
        min_purity=min_purity,
        min_coverage=min_coverage,
        max_vocab_for_full=max_vocab_for_full,
    )
    full_df, _, _ = run_negation_search(
        datapoints,
        mode="full",
        max_size=max_size,
        min_purity=min_purity,
        min_coverage=min_coverage,
        max_vocab_for_full=max_vocab_for_full,
    )

    print(f"Negation analysis: {n_eligible}/{len(datapoints)} eligible "
          f"(no_negation={skipped['no_negation']})\n")

    key_cols = ["run_idx", "run_name", "epoch", "complexity", "properties", "hidden_size", "vocab_size", "n_messages", "n_predicates", "n_neg_predicates"]
    metric_cols = ["status", "max_purity", "max_coverage", "n_pass", "candidates", "best_candidate", "best_purity", "best_coverage"]

    greedy_df = greedy_df.rename(columns={c: f"greedy_{c}" for c in metric_cols})
    full_df = full_df.rename(columns={c: f"full_{c}" for c in metric_cols})

    merged = pd.merge(greedy_df, full_df, on=key_cols, how="outer")
    merged["same_best"] = (
        merged["greedy_best_candidate"].notna()
        & merged["full_best_candidate"].notna()
        & (merged["greedy_best_candidate"] == merged["full_best_candidate"])
    )

    return merged


def run_negation_search(datapoints, *, mode="greedy", max_size=4, min_purity=1.0, min_coverage=1.0, max_vocab_for_full=12):
    '''
    Run a single negation-search mode across all datapoints and return a per-run summary.
    - mode="greedy": heuristic search over disjunctions
    - mode="full": exhaustive (until vocab_size > max_vocab_for_full) disjunction search
    Filters candidates by XOR purity/coverage and records all passing candidates plus the best one.
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

    rows = []
    for run_idx, (datapoint, best_language) in enumerate(tqdm(eligible_runs, desc=f"Negation search ({mode})")):
        config = datapoint.get("config", {})
        language_df = best_language["language"]
        _, vocab = _tokenize_messages(language_df)
        vocab_size = len(vocab)

        # Build candidate disjunctions using one search strategy.
        if mode == "greedy":
            passed, stats = _filter_xor_candidates(
                mi_maximizing_disjunctions_greedy(language_df, max_size=max_size), 
                language_df, min_purity=min_purity, min_coverage=min_coverage
                )
        else:
            if vocab_size <= max_vocab_for_full:
                passed, stats = _filter_xor_candidates(
                    mi_maximizing_disjunctions(language_df, max_size=max_size), 
                    language_df, min_purity=min_purity, min_coverage=min_coverage
                    )
            else:
                passed = pd.DataFrame()
                stats = {"status": "skipped_vocab", "max_purity": None, "max_coverage": None}

        # Keep all passing candidates and note the best one for comparison.
        candidates = ";".join(passed["name"].astype(str).tolist()) if not passed.empty else ""
        best_candidate, best_purity, best_coverage = None, None, None
        if not passed.empty:
            best_row = passed.sort_values(["neg_purity", "neg_coverage"], ascending=False).iloc[0]
            best_candidate = best_row["name"]
            best_purity = float(best_row["neg_purity"])
            best_coverage = float(best_row["neg_coverage"])

        rows.append({
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
            "n_pass": int(len(passed)),
            "candidates": candidates,
            "best_candidate": best_candidate,
            "best_purity": best_purity,
            "best_coverage": best_coverage,
        })

    return pd.DataFrame(rows)

# def symbol_predicate_mi(language: pd.DataFrame):
#     '''Compute mutual information between symbols and predicate identities.'''
#     tokenized, vocab = _tokenize_messages(language)
#     X = _build_presence_matrix(tokenized, vocab)
#     y = language["pred_str"].to_numpy()

#     # MI per single token (bits)
#     mi = [{"token": tok, "mi_bits": mutual_info_score(y, X[:, j]) / np.log(2)} for j, tok in enumerate(vocab)]
#     return pd.DataFrame(mi).sort_values("mi_bits", ascending=False).reset_index(drop=True)

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
        mi_greedy_stats = xor_coverage(toy_set, greedy_mi, item_col="name")
        print(mi_greedy_stats.head(10))
        print()

        print("XOR/coverage stats for exhaustive MI disjunctions:")
        mi_exhaustive_stats = xor_coverage(toy_set, exhaustive_mi, item_col="name")
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
