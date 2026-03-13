"""
analyze.py
----------
Analysis of forced-choice pairwise data.

Analyses:
  1. Participant QC     — RT distributions, left/right bias; configurable exclusion thresholds
  2. Bradley-Terry model — Bayesian importance ranking with participant discriminability
                           random effects; 94% HDIs, posterior rank distributions,
                           pairwise dominance matrix, posterior predictive check
  3. Construct dependence — residual win-rate correlation (heatmap, dendrogram, MDS with
                            variance explained) + binomial test + Bayesian ROPE equivalence
  4. Synonym validity   — hierarchical Bayesian DIF model: tests whether synonym labels
                          shift perceived importance (Binomial-aggregated for efficiency)

Usage:
    python3 analyze.py [--data output] [--out analysis] [OPTIONS]

    QC thresholds (configurable via CLI):
      --speed-threshold MS   Exclude participant if median RT < MS (default 500)
      --side-threshold X     Exclude participant if |left_ratio − 0.5| > X (default 0.2)
      --rt-min MS            Drop trials with RT < MS (default 300)
      --rt-max MS            Drop trials with RT > MS (default 30000)
      --rope-eps X           Equivalence bound for ROPE test (default 0.05)

Dependencies (see requirements.txt):
    pip install pymc arviz  # or use the bundled .venv
"""

import argparse
import warnings
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, binomtest

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

MIN_NONNAN_CORR = 3    # minimum non-NaN correlations for a construct to be clustered
ROPE_EPS_DEFAULT = 0.05  # |P(i>j) - 0.5| < ROPE_EPS → practically interchangeable


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_bt_data(choices: pd.DataFrame):
    """Extract win matrix W, label/ID maps, and trial-level arrays.

    Returns (W, labels, ids, idx_of, n,
             winner_idx, loser_idx, participant_idx, n_participants).
    W[i, j] = number of times item i beat item j (for dependence analysis).
    winner/loser/participant arrays are trial-level (for BT with random effects).
    """
    fc = choices.dropna(subset=["chosen_construct_id", "unchosen_construct_id"]).copy()
    fc["chosen_construct_id"]   = fc["chosen_construct_id"].astype(int)
    fc["unchosen_construct_id"] = fc["unchosen_construct_id"].astype(int)

    id2name = {}
    for _, row in fc.iterrows():
        id2name[int(row["chosen_construct_id"])]   = str(row["chosen_canonical"])
        id2name[int(row["unchosen_construct_id"])] = str(row["unchosen_canonical"])

    ids    = sorted(id2name.keys())
    n      = len(ids)
    idx_of = {cid: i for i, cid in enumerate(ids)}
    labels = [id2name[cid] for cid in ids]

    # Aggregate win matrix
    W = np.zeros((n, n), dtype=int)
    wi_all = fc["chosen_construct_id"].map(idx_of).values
    li_all = fc["unchosen_construct_id"].map(idx_of).values
    for wi, li in zip(wi_all, li_all):
        W[wi, li] += 1

    # Participant index (for random effects)
    participant_ids = sorted(fc["user_id"].unique())
    pid_to_idx      = {pid: i for i, pid in enumerate(participant_ids)}
    participant_idx = fc["user_id"].map(pid_to_idx).values.astype(int)

    return (W, labels, ids, idx_of, n,
            wi_all.astype(int), li_all.astype(int),
            participant_idx, len(participant_ids))


def bayesian_bradley_terry(winner_idx: np.ndarray, loser_idx: np.ndarray,
                            participant_idx: np.ndarray,
                            n_constructs: int, n_participants: int,
                            draws: int = 2000, tune: int = 2000,
                            chains: int = 4):
    """Fit a Bayesian Bradley-Terry model with participant discriminability.

    Construct worth:
        alpha ~ ZeroSumNormal(sigma=1.5)   [log-worth, sum-to-zero constraint]

    Participant discriminability (non-centered parameterisation):
        sigma_kappa    ~ HalfNormal(sigma=0.5)
        kappa_offset[p] ~ Normal(0, 1)
        kappa[p]        = exp(kappa_offset[p] * sigma_kappa)

    Likelihood (one Bernoulli per trial; chosen item always coded as "win"):
        P(i beats j | p) = sigmoid(kappa[p] * (alpha[i] - alpha[j]))

    Convergence checked via R-hat and ESS (warns if R-hat > 1.01).

    Returns (idata, worth_samples, alpha_flat).
    worth_samples has shape (chains*draws, n_constructs).
    alpha_flat    has shape (chains*draws, n_constructs).
    """
    obs = np.ones(len(winner_idx), dtype=np.int8)  # chosen always wins → all 1s

    with pm.Model():
        alpha        = pm.ZeroSumNormal("alpha", sigma=1.5, shape=n_constructs)
        sigma_kappa  = pm.HalfNormal("sigma_kappa", sigma=0.5)
        kappa_offset = pm.Normal("kappa_offset", mu=0, sigma=1, shape=n_participants)
        kappa        = pm.Deterministic("kappa", pm.math.exp(kappa_offset * sigma_kappa))

        log_odds = kappa[participant_idx] * (alpha[winner_idx] - alpha[loser_idx])
        pm.Bernoulli("obs", p=pm.math.sigmoid(log_odds), observed=obs)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=42, progressbar=True)

    # Convergence check (alpha + sigma_kappa)
    summary    = az.summary(idata, var_names=["alpha", "sigma_kappa"])
    max_rhat   = float(summary["r_hat"].max())
    min_ess    = float(summary["ess_bulk"].min())
    kappa_med  = float(np.median(idata.posterior["kappa"].values))
    sigma_k_med = float(np.median(idata.posterior["sigma_kappa"].values))
    if max_rhat > 1.01:
        print(f"  ⚠ WARNING: max R-hat = {max_rhat:.3f} (>1.01) — check convergence!")
    else:
        print(f"  Convergence OK: max R-hat = {max_rhat:.3f}, min ESS = {min_ess:.0f}")
    print(f"  sigma_kappa (discriminability spread): median={sigma_k_med:.3f}  "
          f"kappa median across participants={kappa_med:.3f}")

    alpha_samples = idata.posterior["alpha"].values       # (chains, draws, n)
    alpha_flat    = alpha_samples.reshape(-1, n_constructs)
    a_shifted     = alpha_flat - alpha_flat.max(axis=1, keepdims=True)
    exp_a         = np.exp(a_shifted)
    worth_samples = exp_a / exp_a.sum(axis=1, keepdims=True)

    return idata, worth_samples, alpha_flat


def build_synonym_data(choices: pd.DataFrame):
    """Build Binomial-aggregated data for the hierarchical synonym DIF model.

    Aggregates from trial-level to unique (term_i, term_j) pair level,
    reducing the number of likelihood terms substantially at large N.

    Returns a dict with arrays needed by the PyMC model and label maps.
    """
    needed = ["chosen_construct_id", "unchosen_construct_id",
              "chosen_term", "unchosen_term"]
    fc = choices.dropna(subset=needed).copy()
    for col in ["chosen_construct_id", "unchosen_construct_id"]:
        fc[col] = fc[col].astype(int)

    all_cids   = sorted(set(fc["chosen_construct_id"]) | set(fc["unchosen_construct_id"]))
    cid_to_ci  = {cid: i for i, cid in enumerate(all_cids)}
    cid_to_name = {}
    for _, row in fc.iterrows():
        cid_to_name[int(row["chosen_construct_id"])]   = str(row["chosen_canonical"])
        cid_to_name[int(row["unchosen_construct_id"])] = str(row["unchosen_canonical"])

    # Build term index: unique (construct_id, term_name) pairs
    term_map       = {}
    term_construct = []
    term_is_syn    = []
    term_names     = []

    for _, row in fc.iterrows():
        for cid_col, term_col, syn_col in [
            ("chosen_construct_id",   "chosen_term",   "chosen_is_synonym"),
            ("unchosen_construct_id", "unchosen_term", "unchosen_is_synonym"),
        ]:
            cid  = int(row[cid_col])
            term = str(row[term_col])
            is_s = str(row.get(syn_col, "")).lower() == "true"
            key  = (cid, term)
            if key not in term_map:
                term_map[key] = len(term_map)
                term_construct.append(cid_to_ci[cid])
                term_is_syn.append(is_s)
                term_names.append(term)

    n_terms      = len(term_names)
    n_constructs = len(all_cids)

    # Build trial-level (chosen_ti, unchosen_ti) and aggregate to Binomial
    chosen_ti_all   = []
    unchosen_ti_all = []
    for _, row in fc.iterrows():
        c_cid = int(row["chosen_construct_id"])
        u_cid = int(row["unchosen_construct_id"])
        chosen_ti_all.append(term_map[(c_cid, str(row["chosen_term"]))])
        unchosen_ti_all.append(term_map[(u_cid, str(row["unchosen_term"]))])

    # Aggregate: canonical pair key = (min_ti, max_ti), track wins for min
    from collections import defaultdict
    pair_wins   = defaultdict(int)
    pair_totals = defaultdict(int)
    for c_ti, u_ti in zip(chosen_ti_all, unchosen_ti_all):
        lo, hi = min(c_ti, u_ti), max(c_ti, u_ti)
        pair_totals[(lo, hi)] += 1
        if c_ti == lo:     # lo was chosen → win for lo
            pair_wins[(lo, hi)] += 1

    agg_i      = np.array([k[0] for k in pair_totals])
    agg_j      = np.array([k[1] for k in pair_totals])
    agg_wins   = np.array([pair_wins[k] for k in pair_totals], dtype=int)
    agg_totals = np.array([pair_totals[k] for k in pair_totals], dtype=int)

    # Synonym DIF index mapping
    term_is_syn_arr  = np.array(term_is_syn, dtype=bool)
    syn_term_indices = np.where(term_is_syn_arr)[0]
    n_synonyms       = len(syn_term_indices)
    syn_delta_idx    = np.full(n_terms, n_synonyms, dtype=int)  # sentinel for canonicals
    for di, ti in enumerate(syn_term_indices):
        syn_delta_idx[ti] = di

    return dict(
        agg_i            = agg_i,
        agg_j            = agg_j,
        agg_wins         = agg_wins,
        agg_totals       = agg_totals,
        term_construct   = np.array(term_construct),
        term_is_syn      = term_is_syn_arr,
        syn_term_indices = syn_term_indices,
        syn_delta_idx    = syn_delta_idx,
        n_synonyms       = n_synonyms,
        n_terms          = n_terms,
        n_constructs     = n_constructs,
        term_names       = term_names,
        all_cids         = all_cids,
        cid_to_name      = cid_to_name,
        term_construct_list = term_construct,
    )


def classical_mds(D: np.ndarray, k: int = 2):
    """Project a symmetric distance matrix to k dimensions via classical MDS.

    Returns (coords, var_explained) where var_explained is the proportion of
    total positive variance captured by the first k dimensions.
    """
    n = D.shape[0]
    D2 = np.square(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D2 @ J)
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    pos_eig    = np.maximum(eigenvalues, 0.0)
    coords     = eigenvectors[:, :k] * np.sqrt(pos_eig[:k])
    var_explained = pos_eig[:k].sum() / pos_eig.sum() if pos_eig.sum() > 0 else 0.0
    return coords, var_explained


def apply_qc_filters(choices: pd.DataFrame, qc: pd.DataFrame,
                     rt_min_ms: float = 300,
                     rt_max_ms: float = 30_000) -> pd.DataFrame:
    """Exclude flagged participants (flag_speed OR flag_side) and out-of-range trials."""
    bad_pids = set(qc.loc[qc["flag_speed"] | qc["flag_side"], "user_id"])
    if bad_pids:
        print(f"  Excluding {len(bad_pids)} flagged participant(s): "
              f"{', '.join(str(p)[:8] for p in bad_pids)}")
    else:
        print("  No participants flagged for exclusion.")

    filtered = choices[~choices["user_id"].isin(bad_pids)].copy()
    rt       = filtered["response_time"].astype(float)
    n_before = len(filtered)
    filtered = filtered[(rt >= rt_min_ms) & (rt <= rt_max_ms)].copy()
    n_dropped = n_before - len(filtered)
    if n_dropped:
        print(f"  Dropped {n_dropped} trials outside RT [{rt_min_ms:.0f}, {rt_max_ms:.0f}] ms.")

    print(f"  Retained {filtered['user_id'].nunique()} participants, "
          f"{len(filtered)} trials after QC filtering.")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# 1. Participant QC
# ─────────────────────────────────────────────────────────────────────────────

def participant_qc(choices: pd.DataFrame, attn: pd.DataFrame, out: Path,
                   speed_threshold_ms: float = 500,
                   side_threshold: float = 0.2) -> pd.DataFrame:
    """Compute per-participant QC metrics and flags.

    Flags (saved to CSV; exclusion applied separately via apply_qc_filters):
      flag_speed : median RT < speed_threshold_ms  (speed-clicking)
      flag_side  : |left_ratio − 0.5| > side_threshold  (side bias)

    Attention check results are recorded for reporting but not used as an
    exclusion criterion — participants who fail attention checks are terminated
    by the experiment before completing the session and therefore do not appear
    in the analysis dataset.
    """
    print(f"\n=== 1. Participant QC ===")
    print(f"  Thresholds: speed<{speed_threshold_ms}ms, "
          f"side>{side_threshold:.2f} from 0.5")
    participants = []
    for pid, g in choices.groupby("user_id"):
        n_trials   = len(g)
        rt         = g["response_time"].astype(float)
        n_left     = (g["chosen_term"] == g["left_term"]).sum()
        left_ratio = n_left / n_trials if n_trials > 0 else np.nan

        ac     = attn[attn["user_id"] == pid]
        n_attn = len(ac)
        n_pass = (ac["passed"].astype(str).str.lower() == "true").sum() if n_attn > 0 else 0

        participants.append({
            "user_id":       pid,
            "n_trials":      n_trials,
            "rt_median_s":   round(rt.median() / 1000, 2),
            "rt_p5_s":       round(rt.quantile(0.05) / 1000, 2),
            "rt_p95_s":      round(rt.quantile(0.95) / 1000, 2),
            "left_ratio":    round(left_ratio, 3),
            "n_attn_checks": n_attn,
            "n_attn_passed": n_pass,
            "flag_speed":    rt.median() < speed_threshold_ms,
            "flag_side":     abs(left_ratio - 0.5) > side_threshold,
        })

    qc = pd.DataFrame(participants)
    print(qc[["user_id", "n_trials", "rt_median_s", "left_ratio",
              "n_attn_checks", "n_attn_passed", "flag_speed", "flag_side"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(qc)))

    for (pid, g), col in zip(choices.groupby("user_id"), colors):
        rt_s = g["response_time"].astype(float) / 1000
        axes[0].hist(rt_s.clip(0, 30), bins=40, alpha=0.5, label=pid[:8], color=col, density=True)
    axes[0].set_xlabel("Response time (s)");  axes[0].set_ylabel("Density")
    axes[0].set_title("RT distributions per participant")
    axes[0].set_xlim(0, 30);  axes[0].legend(fontsize=7, ncol=2)

    x = np.arange(len(qc))
    bar_colors = ["tomato" if f else "steelblue" for f in qc["flag_side"]]
    axes[1].bar(x, qc["left_ratio"], color=bar_colors)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.5 - side_threshold, color="red", linestyle=":", linewidth=1)
    axes[1].axhline(0.5 + side_threshold, color="red", linestyle=":", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p[:8] for p in qc["user_id"]], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Proportion choosing LEFT")
    axes[1].set_title("Left/right ratio  (red = flagged)")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    p = out / "1_participant_qc.png"
    fig.savefig(p);  plt.close(fig)
    print(f"  → {p}")

    qc.to_csv(out / "1_participant_qc.csv", index=False)
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bradley-Terry model (Bayesian via PyMC, with participant random effects)
# ─────────────────────────────────────────────────────────────────────────────

def bradley_terry_analysis(choices: pd.DataFrame, out: Path):
    print("\n=== 2. Bradley-Terry Model (Bayesian + participant random effects) ===")

    (W, labels, ids, idx_of, n,
     winner_idx, loser_idx, participant_idx, n_participants) = build_bt_data(choices)

    print(f"  Sampling posterior for {n} constructs, {n_participants} participants …")
    idata, worth_samples, alpha_flat = bayesian_bradley_terry(
        winner_idx, loser_idx, participant_idx, n, n_participants)
    n_samples = worth_samples.shape[0]

    worth_median = np.median(worth_samples, axis=0)
    hdi_bounds   = az.hdi(worth_samples, hdi_prob=0.94)
    hdi_lo, hdi_hi = hdi_bounds[:, 0], hdi_bounds[:, 1]

    alpha_median  = np.median(alpha_flat, axis=0)
    alpha_hdi_arr = az.hdi(alpha_flat, hdi_prob=0.94)   # (n, 2)
    alpha_hdi_lo  = alpha_hdi_arr[:, 0]
    alpha_hdi_hi  = alpha_hdi_arr[:, 1]

    rank_samples = np.argsort(np.argsort(-worth_samples, axis=1), axis=1)
    rank_hdi_arr = az.hdi(rank_samples.astype(float), hdi_prob=0.94)
    rank_median  = np.median(rank_samples, axis=0).astype(int)

    order = np.argsort(worth_median)[::-1]
    ranked = [
        (i + 1, labels[j],
         round(worth_median[j], 5), round(hdi_lo[j], 5), round(hdi_hi[j], 5),
         int(rank_median[j]) + 1,
         int(rank_hdi_arr[j, 0]) + 1, int(rank_hdi_arr[j, 1]) + 1,
         int(ids[j]))
        for i, j in enumerate(order)
    ]
    print("  Top 10:")
    for rank, name, w, lo, hi, rmed, rlo, rhi, cid in ranked[:10]:
        print(f"    {rank:3d}. {name:<40s}  worth={w:.4f}  "
              f"94%HDI=[{lo:.4f},{hi:.4f}]  rank_HDI=[{rlo},{rhi}]")

    df_rank = pd.DataFrame(ranked, columns=[
        "rank", "construct", "bt_worth_median", "hdi94_lo", "hdi94_hi",
        "rank_median", "rank_hdi94_lo", "rank_hdi94_hi", "construct_id",
    ])
    df_rank["alpha_median"]   = [round(float(alpha_median[j]),  5) for j in order]
    df_rank["alpha_hdi94_lo"] = [round(float(alpha_hdi_lo[j]),  5) for j in order]
    df_rank["alpha_hdi94_hi"] = [round(float(alpha_hdi_hi[j]),  5) for j in order]
    df_rank.to_csv(out / "2_bt_rankings.csv", index=False)

    # ── Plot 2a: ranked bar chart + HDI ──
    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.18)))
    y        = np.arange(n)
    s_worth  = worth_median[order][::-1]
    s_labels = [labels[j] for j in order][::-1]
    s_lo     = hdi_lo[order][::-1]
    s_hi     = hdi_hi[order][::-1]
    ax.barh(y, s_worth, color="steelblue", alpha=0.8)
    ax.errorbar(s_worth, y,
                xerr=[s_worth - s_lo, s_hi - s_worth],
                fmt="none", color="black", linewidth=0.7, capsize=2)
    ax.set_yticks(y);  ax.set_yticklabels(s_labels, fontsize=7)
    ax.set_xlabel("Bradley-Terry worth (posterior median)  ±94% HDI")
    ax.set_title(f"Construct importance ranking  "
                 f"(N={choices['user_id'].nunique()} participants, with discriminability RE)")
    plt.tight_layout()
    p = out / "2a_bt_rankings.png"
    fig.savefig(p, bbox_inches="tight");  plt.close(fig)
    print(f"  → {p}")

    # ── Plot 2b: posterior rank distributions ──
    rank_prob = np.zeros((n, n))
    for s in range(n_samples):
        rank_prob[np.arange(n), rank_samples[s]] += 1
    rank_prob /= n_samples

    fig, ax = plt.subplots(figsize=(max(8, n * 0.15), max(6, n * 0.15)))
    im = ax.imshow(rank_prob[order, :], aspect="auto",
                   cmap="YlOrRd", vmin=0, interpolation="nearest")
    ax.set_xlabel("Rank position (1 = most important)")
    ax.set_ylabel("Construct  (ordered by posterior median worth)")
    ax.set_yticks(range(n));  ax.set_yticklabels([labels[j] for j in order], fontsize=5)
    ax.set_xticks(range(0, n, max(1, n // 10)))
    ax.set_xticklabels(range(1, n + 1, max(1, n // 10)), fontsize=7)
    ax.set_title("Posterior rank distribution  P(rank = k | data)")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Posterior probability")
    plt.tight_layout()
    p = out / "2b_bt_rank_distributions.png"
    fig.savefig(p, bbox_inches="tight");  plt.close(fig)
    print(f"  → {p}")

    # ── Plot 2c: pairwise dominance matrix ──
    dominance   = (worth_samples[:, :, None] > worth_samples[:, None, :]).mean(axis=0)
    dom_ordered = dominance[np.ix_(order, order)]
    olabels     = [labels[j] for j in order]

    fig, ax = plt.subplots(figsize=(max(7, n * 0.12), max(6, n * 0.12)))
    im = ax.imshow(dom_ordered, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n));  ax.set_xticklabels(olabels, rotation=90, fontsize=4)
    ax.set_yticks(range(n));  ax.set_yticklabels(olabels, fontsize=4)
    ax.set_title("Pairwise dominance  P(row worth > col worth | data)")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    p = out / "2c_bt_dominance.png"
    fig.savefig(p, bbox_inches="tight");  plt.close(fig)
    print(f"  → {p}")

    # ── Plot 2d: posterior predictive check (full model, kappa marginalised) ──
    # For each pair (i,j) and posterior sample s:
    #   p_pred_s = mean_p sigmoid(kappa_p_s * (alpha_s[i] - alpha_s[j]))
    # This faithfully reflects the full model likelihood, including participant-
    # level discriminability variability, by averaging over all participants.
    pairs_i_ppc, pairs_j_ppc, obs_rate_ppc = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            total = int(W[i, j] + W[j, i])
            if total == 0:
                continue
            pairs_i_ppc.append(i);  pairs_j_ppc.append(j)
            obs_rate_ppc.append(W[i, j] / total)
    pairs_i_ppc  = np.array(pairs_i_ppc)
    pairs_j_ppc  = np.array(pairs_j_ppc)
    obs_rate_ppc = np.array(obs_rate_ppc)

    kappa_flat = idata.posterior["kappa"].values.reshape(-1, n_participants)  # (S, P)
    ppc_mean   = np.empty(len(pairs_i_ppc))
    ppc_lo     = np.empty(len(pairs_i_ppc))
    ppc_hi     = np.empty(len(pairs_i_ppc))
    for k, (pi, pj) in enumerate(zip(pairs_i_ppc, pairs_j_ppc)):
        diff_s       = alpha_flat[:, pi] - alpha_flat[:, pj]   # (S,)
        logits       = kappa_flat * diff_s[:, None]             # (S, P)
        p_per_sample = 1.0 / (1.0 + np.exp(-logits)).mean(axis=1)  # (S,)
        ppc_mean[k]  = p_per_sample.mean()
        ppc_lo[k]    = np.percentile(p_per_sample,  3)
        ppc_hi[k]    = np.percentile(p_per_sample, 97)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(obs_rate_ppc, ppc_mean,
                yerr=[np.maximum(0, ppc_mean - ppc_lo),
                      np.maximum(0, ppc_hi   - ppc_mean)],
                fmt="o", alpha=0.3, markersize=3, linewidth=0.5, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Observed win rate")
    ax.set_ylabel("Posterior predictive win rate  (mean ± 94% interval)")
    ax.set_title("Posterior predictive check: per-pair win rates\n"
                 "(predictions marginalised over participant discriminability)")
    ax.set_xlim(-0.05, 1.05);  ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    p = out / "2d_bt_ppc.png"
    fig.savefig(p, bbox_inches="tight");  plt.close(fig)
    print(f"  → {p}")

    return worth_median, worth_samples, alpha_median, labels, W, ids, idx_of, order


# ─────────────────────────────────────────────────────────────────────────────
# 3. Construct dependence — residual correlation + ROPE equivalence
# ─────────────────────────────────────────────────────────────────────────────

def construct_dependence(worth_samples: np.ndarray, worth_median: np.ndarray,
                         alpha_median: np.ndarray,
                         labels: list, W: np.ndarray, ids, idx_of,
                         out: Path, rope_eps: float = ROPE_EPS_DEFAULT):
    print("\n=== 3. Construct Dependence ===")
    n = len(worth_median)

    # Residual win-rate correlation (Spearman, using posterior median for P_pred)
    P_obs  = np.full((n, n), np.nan)
    P_pred = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total = W[i, j] + W[j, i]
            if total > 0:
                P_obs[i, j]  = W[i, j] / total
                P_pred[i, j] = 1.0 / (1.0 + np.exp(-(alpha_median[i] - alpha_median[j])))
    residuals = P_obs - P_pred

    dep = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            mask = ~np.isnan(residuals[i]) & ~np.isnan(residuals[j])
            if mask.sum() < 3:
                continue
            r, _ = spearmanr(residuals[i][mask], residuals[j][mask])
            dep[i, j] = dep[j, i] = r
    np.fill_diagonal(dep, 1.0)

    dep_filled = np.where(np.isnan(dep), 0.0, dep)

    n_nonnan = (~np.isnan(dep)).sum(axis=1) - 1
    keep     = n_nonnan >= MIN_NONNAN_CORR
    n_keep   = keep.sum()
    print(f"  {n_keep}/{n} constructs have ≥{MIN_NONNAN_CORR} non-NaN residual correlations "
          f"(used for clustering/MDS).")

    dep_sub    = dep_filled[np.ix_(keep, keep)]
    labels_sub = [labels[i] for i in range(n) if keep[i]]
    dist_sub   = np.clip(1 - dep_sub, 0, None)
    np.fill_diagonal(dist_sub, 0)
    linkage_matrix = linkage(squareform(dist_sub), method="average")

    fig, axes = plt.subplots(1, 3, figsize=(26, max(6, n * 0.15)))

    # Heatmap
    dep_masked = np.ma.array(dep, mask=np.isnan(dep))
    cmap = plt.cm.RdBu_r.copy();  cmap.set_bad("lightgray")
    im = axes[0].imshow(dep_masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels(labels, rotation=90, fontsize=5)
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels(labels, fontsize=5)
    axes[0].set_title("Residual win-rate correlation\n(Spearman; gray = no data)")
    plt.colorbar(im, ax=axes[0], shrink=0.6)

    # Dendrogram
    dendrogram(linkage_matrix, ax=axes[1], labels=labels_sub,
               orientation="right", leaf_font_size=7)
    axes[1].set_title(f"Hierarchical clustering\n(n={n_keep} constructs with data)")

    # MDS with variance explained
    coords, var_exp = classical_mds(dist_sub, k=2)
    axes[2].scatter(coords[:, 0], coords[:, 1], s=20, color="steelblue", zorder=3)
    for k, (x, y_coord) in enumerate(coords):
        axes[2].annotate(labels_sub[k], (x, y_coord), fontsize=5,
                         ha="center", va="bottom", xytext=(0, 3),
                         textcoords="offset points")
    axes[2].set_title(f"MDS map of construct neighbourhoods\n"
                      f"(Spearman distance; dim1+dim2 = {var_exp:.1%} of variance)")
    axes[2].set_xlabel("MDS dim 1");  axes[2].set_ylabel("MDS dim 2")
    axes[2].axhline(0, color="lightgray", linewidth=0.5)
    axes[2].axvline(0, color="lightgray", linewidth=0.5)

    plt.tight_layout()
    p = out / "3_construct_dependence.png"
    fig.savefig(p, bbox_inches="tight");  plt.close(fig)
    print(f"  → {p}")

    # Highly correlated pairs
    print("  Highly correlated construct pairs (Spearman r > 0.6):")
    found = False
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(dep[i, j]) and dep[i, j] > 0.6:
                print(f"    r={dep[i,j]:.3f}  {labels[i]}  ↔  {labels[j]}")
                found = True
    if not found:
        print("    (none)")

    # ── Bayesian ROPE equivalence test ──
    # P(|P(i beats j | data) - 0.5| < rope_eps) using the full worth posterior.
    # A pair is declared practically interchangeable when this probability ≥ 0.95.
    print(f"\n  Bayesian ROPE equivalence test  (ε={rope_eps}, threshold P≥0.95):")
    print(f"  Pairs where constructs are practically interchangeable "
          f"(win probability within [{0.5 - rope_eps:.2f}, {0.5 + rope_eps:.2f}]):")
    rope_rows = []
    found_rope = False
    for i in range(n):
        for j in range(i + 1, n):
            total = int(W[i, j] + W[j, i])
            if total < 20:
                continue
            p_ij      = worth_samples[:, i] / (worth_samples[:, i] + worth_samples[:, j])
            rope_prob = float(np.mean(np.abs(p_ij - 0.5) < rope_eps))
            if rope_prob >= 0.95:
                wr = W[i, j] / total
                print(f"    P(ROPE)={rope_prob:.3f}  obs_wr={wr:.2f}  "
                      f"{labels[i]}  ↔  {labels[j]}")
                rope_rows.append({
                    "construct_a": labels[i], "construct_b": labels[j],
                    "n_comparisons": total, "obs_win_rate_a": round(wr, 3),
                    "rope_prob": round(rope_prob, 4),
                })
                found_rope = True
    if not found_rope:
        print(f"    (no pairs meet the threshold at ε={rope_eps})")

    if rope_rows:
        pd.DataFrame(rope_rows).to_csv(out / "3_rope_equivalence.csv", index=False)
        print(f"  → {out / '3_rope_equivalence.csv'}")

    # Binomial test (secondary check)
    print("\n  Secondary: near-chance pairs (binomial p > 0.10 vs 50/50):")
    found_chance = False
    for i in range(n):
        for j in range(i + 1, n):
            total = int(W[i, j] + W[j, i])
            if total < 10:
                continue
            wins_i = int(W[i, j])
            result = binomtest(wins_i, total, 0.5)
            if result.pvalue > 0.10:
                print(f"    p={result.pvalue:.3f}  wr={wins_i/total:.2f}  "
                      f"{labels[i]}  ↔  {labels[j]}")
                found_chance = True
    if not found_chance:
        print("    (none)")

    pd.DataFrame(dep, index=labels, columns=labels).to_csv(out / "3_dependence_matrix.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Synonym validity — hierarchical Bayesian DIF model (Binomial-aggregated)
# ─────────────────────────────────────────────────────────────────────────────

def synonym_dif_analysis(choices: pd.DataFrame, out: Path,
                          draws: int = 2000, tune: int = 2000, chains: int = 4):
    """Fit a hierarchical Bradley-Terry DIF model at the term level.

    Model (Binomial-aggregated over unique term pairs for efficiency):
        alpha_construct[c] ~ ZeroSumNormal(sigma=1.5)
        sigma_dif          ~ HalfNormal(sigma=0.5)
        delta_syn[s]       ~ Normal(0, sigma_dif)   [synonym DIF; 0 for canonical]
        alpha_term[t]      = alpha_construct[c(t)] + delta_syn[s(t)]
        wins_ij            ~ Binomial(total_ij, sigmoid(alpha_term[i] - alpha_term[j]))

    A synonym with 94% HDI of delta excluding 0 indicates the label itself
    shifts perceived importance beyond the underlying construct.
    sigma_dif summarises the global magnitude of label effects.
    """
    print("\n=== 4. Synonym Validity (Hierarchical DIF Model) ===")

    d = build_synonym_data(choices)
    if d["n_synonyms"] == 0:
        print("  No synonym terms found in data.")
        return

    n_pairs = len(d["agg_i"])
    print(f"  {d['n_constructs']} constructs, {d['n_terms']} unique terms "
          f"({d['n_synonyms']} synonyms), {n_pairs} unique term pairs.")

    term_construct = d["term_construct"]
    syn_delta_idx  = d["syn_delta_idx"]
    n_synonyms     = d["n_synonyms"]

    print(f"  Sampling DIF posterior for {n_synonyms} synonym terms …")
    with pm.Model():
        alpha_c   = pm.ZeroSumNormal("alpha_construct", sigma=1.5, shape=d["n_constructs"])
        sigma_dif = pm.HalfNormal("sigma_dif", sigma=0.5)
        delta_syn = pm.Normal("delta_syn", mu=0, sigma=sigma_dif, shape=n_synonyms)

        delta_with_zero = pt.concatenate([delta_syn, pt.zeros(1)])
        alpha_term_full = alpha_c[term_construct] + delta_with_zero[syn_delta_idx]

        p = pm.math.sigmoid(alpha_term_full[d["agg_i"]] - alpha_term_full[d["agg_j"]])
        pm.Binomial("obs", n=d["agg_totals"], p=p, observed=d["agg_wins"])

        idata_dif = pm.sample(draws=draws, tune=tune, chains=chains,
                              target_accept=0.9, random_seed=42, progressbar=True)

    # Convergence
    summary_dif = az.summary(idata_dif, var_names=["delta_syn", "sigma_dif"])
    max_rhat    = float(summary_dif["r_hat"].max())
    min_ess     = float(summary_dif["ess_bulk"].min())
    if max_rhat > 1.01:
        print(f"  ⚠ WARNING: max R-hat = {max_rhat:.3f} — check DIF convergence!")
    else:
        print(f"  DIF convergence OK: max R-hat = {max_rhat:.3f}, min ESS = {min_ess:.0f}")

    sigma_samples = idata_dif.posterior["sigma_dif"].values.flatten()
    sigma_hdi     = az.hdi(sigma_samples, hdi_prob=0.94)
    print(f"  sigma_dif: median={np.median(sigma_samples):.3f}, "
          f"94%HDI=[{sigma_hdi[0]:.3f}, {sigma_hdi[1]:.3f}]  "
          f"(global scale of label effects on log-odds scale)")

    delta_samples = idata_dif.posterior["delta_syn"].values.reshape(-1, n_synonyms)
    delta_median  = np.median(delta_samples, axis=0)
    delta_hdi     = az.hdi(delta_samples, hdi_prob=0.94)
    delta_lo, delta_hi = delta_hdi[:, 0], delta_hdi[:, 1]
    sig_dif       = (delta_lo > 0) | (delta_hi < 0)

    syn_ti = d["syn_term_indices"]
    rows = []
    for di, ti in enumerate(syn_ti):
        cid   = d["all_cids"][d["term_construct_list"][ti]]
        cname = d["cid_to_name"].get(cid, str(cid))
        rows.append({
            "construct_id": cid,
            "canonical":    cname,
            "synonym_term": d["term_names"][ti],
            "delta_median": round(float(delta_median[di]), 4),
            "hdi94_lo":     round(float(delta_lo[di]), 4),
            "hdi94_hi":     round(float(delta_hi[di]), 4),
            "sig_dif":      bool(sig_dif[di]),
        })

    df_dif = pd.DataFrame(rows).sort_values("delta_median", ascending=False)
    n_sig  = df_dif["sig_dif"].sum()
    print(f"\n  {n_sig}/{len(df_dif)} synonyms show significant DIF (94% HDI excludes 0):")
    print(df_dif[["canonical", "synonym_term", "delta_median",
                  "hdi94_lo", "hdi94_hi", "sig_dif"]].to_string(index=False))
    df_dif.to_csv(out / "4_synonym_dif.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(6, len(df_dif) * 0.45), 5))
    x = np.arange(len(df_dif))
    bar_colors = ["tomato" if s else "steelblue" for s in df_dif["sig_dif"]]
    ax.bar(x, df_dif["delta_median"], color=bar_colors, alpha=0.8)
    ax.errorbar(x, df_dif["delta_median"],
                yerr=[df_dif["delta_median"] - df_dif["hdi94_lo"],
                      df_dif["hdi94_hi"] - df_dif["delta_median"]],
                fmt="none", color="black", linewidth=0.8, capsize=3)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['canonical'][:16]}\n↔ {row['synonym_term'][:16]}"
         for _, row in df_dif.iterrows()],
        rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("DIF offset δ  (positive = synonym perceived as more important)")
    ax.set_title(f"Synonym DIF analysis  (red = 94% HDI excludes 0;  "
                 f"σ_dif median={np.median(sigma_samples):.3f})")
    plt.tight_layout()
    p = out / "4_synonym_dif.png"
    fig.savefig(p, bbox_inches="tight");  plt.close(fig)
    print(f"  → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze forced-choice pairwise data.")
    parser.add_argument("--data", default="output",
                        help="Directory with choice_data_*.csv files")
    parser.add_argument("--out",  default="analysis",
                        help="Output directory for plots and tables")
    parser.add_argument("--speed-threshold", type=float, default=500, metavar="MS",
                        help="Exclude participant if median RT < MS (default 500)")
    parser.add_argument("--side-threshold",  type=float, default=0.2,  metavar="X",
                        help="Exclude participant if |left_ratio − 0.5| > X (default 0.2)")
    parser.add_argument("--rt-min", type=float, default=300,    metavar="MS",
                        help="Drop trials with RT < MS (default 300)")
    parser.add_argument("--rt-max", type=float, default=30_000, metavar="MS",
                        help="Drop trials with RT > MS (default 30000)")
    parser.add_argument("--rope-eps", type=float, default=ROPE_EPS_DEFAULT, metavar="X",
                        help=f"ROPE equivalence bound (default {ROPE_EPS_DEFAULT})")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir   = (script_dir / args.data).resolve()
    out_dir    = (script_dir / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir : {data_dir}")
    print(f"Output   : {out_dir}")

    demo_rows, choice_rows, attn_rows = [], [], []
    for csv_file in sorted(data_dir.glob("choice_data_*.csv")):
        df = pd.read_csv(csv_file, dtype=str)
        demo_rows.append(df[df["task"] == "demographics"])
        choice_rows.append(df[df["task"] == "forced_choice"])
        attn_rows.append(df[df["task"] == "attention_check"])

    demo    = pd.concat(demo_rows,   ignore_index=True) if demo_rows   else pd.DataFrame()
    choices = pd.concat(choice_rows, ignore_index=True) if choice_rows else pd.DataFrame()
    attn    = pd.concat(attn_rows,   ignore_index=True) if attn_rows   else pd.DataFrame()

    for col in ["trial_index", "left_construct_id", "right_construct_id",
                "chosen_construct_id", "unchosen_construct_id",
                "response_time", "subset_index"]:
        if col in choices.columns:
            choices[col] = pd.to_numeric(choices[col], errors="coerce")

    print(f"\nLoaded {choices['user_id'].nunique()} participants, "
          f"{len(choices)} forced-choice trials, "
          f"{len(attn)} attention-check trials.")

    if choices.empty:
        print("No forced-choice data found. Exiting.")
        return

    # 1. QC
    qc = participant_qc(choices, attn, out_dir,
                        speed_threshold_ms=args.speed_threshold,
                        side_threshold=args.side_threshold)

    print("\n--- Applying QC exclusions ---")
    choices_clean = apply_qc_filters(choices, qc,
                                     rt_min_ms=args.rt_min,
                                     rt_max_ms=args.rt_max)
    if choices_clean.empty:
        print("No data remaining after QC filtering. Exiting.")
        return

    # 2. BT (with participant random effects)
    worth_median, worth_samples, alpha_median, labels, W, ids, idx_of, order = \
        bradley_terry_analysis(choices_clean, out_dir)

    # 3. Dependence + ROPE
    construct_dependence(worth_samples, worth_median, alpha_median, labels, W, ids, idx_of,
                         out_dir, rope_eps=args.rope_eps)

    # 4. Synonym DIF
    synonym_dif_analysis(choices_clean, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
