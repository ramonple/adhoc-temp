"""
approval_rules.py

End-to-end utilities for:
1) Best-tail analysis (feature screening for approval rules)
2) Config dataframe construction (row-by-row)
3) Grid-search approval rule generation (1D / 2D / 3D) with min/max bounds
4) Example usage

Assumptions:
- bad_flag is 0/1 (1 = bad, 0 = good)
- Numerical features are in num_list
- direction convention (from your config):
    direction =  1  -> higher values are riskier (approve LOW values)
    direction = -1  -> lower values are riskier (approve HIGH values)
"""

from __future__ import annotations

import itertools
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 1) Best-tail analysis (approval feature screening)
# =============================================================================

def _clean_binary_target(df: pd.DataFrame, bad_flag: str) -> pd.DataFrame:
    """Keep only rows where bad_flag is 0/1 and cast to int."""
    y = pd.to_numeric(df[bad_flag], errors="coerce")
    out = df.loc[y.isin([0, 1])].copy()
    out[bad_flag] = y.loc[out.index].astype(int)
    return out


def _normalize_data_dictionary(data_dictionary: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Normalize columns to lower-case and require: variable, definition, direction.
    Returns a copy with columns lowercased, or None.
    """
    if data_dictionary is None:
        return None

    dd = data_dictionary.copy()
    dd.columns = [c.strip().lower() for c in dd.columns]
    required = {"variable", "definition", "direction"}
    if not required.issubset(set(dd.columns)):
        raise ValueError(
            f"data_dictionary must include columns: {required} (case-insensitive). Got: {set(dd.columns)}"
        )
    dd["variable"] = dd["variable"].astype(str).str.strip()
    return dd


def _clean_variable_name(target_variable: str) -> str:
    """
    Match your prior convention:
      - drop prefix up to last underscore
      - remove trailing ' -' patterns
    """
    s = str(target_variable)
    cleaned = s.split("_")[-1] if "_" in s else s
    cleaned = cleaned.rstrip(" -") if " -" in cleaned else cleaned
    return cleaned


def rank_features_by_best_tail_good_vol(
    data: pd.DataFrame,
    bad_flag: str,
    num_list: list[str],
    data_dictionary: Optional[pd.DataFrame] = None,
    best_pct: float = 0.05,           # best 5% by default
    min_non_missing: int = 200,
    missing_sentinels: tuple = (-9999,),
    direction_default: int = 0,       # 0 => infer via Spearman if missing
) -> pd.DataFrame:
    """
    Rank numerical features by GOOD VOLUME captured in the best X% tail.

    Best-tail definition (mirror of worst-tail):
      - if direction_used == 1 (high values are riskier), best tail = bottom X%
      - if direction_used == -1 (low values are riskier), best tail = top X%
      - if direction_used == 0, infer via Spearman corr(feature, bad_flag)
    """
    if not (0 < best_pct < 1):
        raise ValueError("best_pct must be between 0 and 1 (e.g., 0.05 for 5%).")

    df = _clean_binary_target(data.copy(), bad_flag)
    overall_bad_rate = df[bad_flag].mean()
    if pd.isna(overall_bad_rate):
        raise ValueError("Overall bad rate is NaN (bad_flag might be empty after cleaning).")
    overall_good_rate = 1 - overall_bad_rate

    dd = _normalize_data_dictionary(data_dictionary)
    dd_lookup = dd.set_index("variable", drop=False) if dd is not None else None

    results: list[dict] = []

    for col in num_list:
        if col not in df.columns:
            continue

        s_raw = pd.to_numeric(df[col], errors="coerce")
        for ms in missing_sentinels:
            s_raw = s_raw.mask(s_raw == ms)

        mask = s_raw.notna()
        if int(mask.sum()) < min_non_missing:
            continue

        x = s_raw.loc[mask]
        y = df.loc[mask, bad_flag]

        cleaned_name = _clean_variable_name(col)

        # dictionary info
        definition = None
        direction = direction_default
        if dd_lookup is not None and cleaned_name in dd_lookup.index:
            row = dd_lookup.loc[cleaned_name]
            definition = row.get("definition", None)
            d = row.get("direction", np.nan)
            if pd.notnull(d):
                try:
                    direction = int(float(d))
                except Exception:
                    direction = direction_default

        inferred_corr = np.nan
        direction_used = direction
        if direction_used == 0:
            inferred_corr = x.corr(y, method="spearman")
            if pd.isna(inferred_corr) or inferred_corr == 0:
                direction_used = 1  # fallback
            else:
                direction_used = 1 if inferred_corr > 0 else -1

        # best tail selection
        if direction_used == 1:
            # high is risky => best tail is bottom X%
            cutoff = float(x.quantile(best_pct))
            tail_mask = x <= cutoff
            tail_side = f"bottom_{int(best_pct*100)}%"
        else:
            # low is risky => best tail is top X%
            cutoff = float(x.quantile(1 - best_pct))
            tail_mask = x >= cutoff
            tail_side = f"top_{int(best_pct*100)}%"

        tail_n = int(tail_mask.sum())
        if tail_n == 0:
            continue

        tail_bad_n = int(y.loc[tail_mask].sum())
        tail_good_n = int(tail_n - tail_bad_n)
        tail_bad_rate = tail_bad_n / tail_n
        tail_good_rate = 1 - tail_bad_rate
        good_rate_lift = tail_good_rate / overall_good_rate if overall_good_rate > 0 else np.nan

        results.append({
            "feature": col,
            "cleaned_name": cleaned_name,
            "definition": definition,
            "direction_used": direction_used,
            "tail_side": tail_side,
            "cutoff": cutoff,

            "tail_n": tail_n,
            "tail_bad_n": tail_bad_n,
            "tail_good_n": tail_good_n,
            "tail_bad_rate": tail_bad_rate,
            "tail_good_rate": tail_good_rate,

            "overall_bad_rate": overall_bad_rate,
            "overall_good_rate": overall_good_rate,
            "good_rate_lift": good_rate_lift,

            "spearman_corr_if_inferred": inferred_corr
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Approval feature screening: lowest tail bad rate first, then larger good volume
    out = out.sort_values(
        ["tail_bad_rate", "tail_good_n", "tail_n"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def rank_features_by_best_tail_good_bal(
    data: pd.DataFrame,
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
    num_list: list[str],
    data_dictionary: Optional[pd.DataFrame] = None,
    best_pct: float = 0.05,
    min_non_missing: int = 200,
    missing_sentinels: tuple = (-9999,),
    direction_default: int = 0,
) -> pd.DataFrame:
    """
    Rank numerical features by GOOD BALANCE captured in the best X% tail.

    good_balance := total_bal - bad_bal
    """
    if not (0 < best_pct < 1):
        raise ValueError("best_pct must be between 0 and 1 (e.g., 0.05 for 5%).")

    df = _clean_binary_target(data.copy(), bad_flag)

    df[total_bal] = pd.to_numeric(df[total_bal], errors="coerce")
    df[bad_bal] = pd.to_numeric(df[bad_bal], errors="coerce")
    good_bal_all = df[total_bal] - df[bad_bal]
    total_good_balance_all = float(good_bal_all.sum(skipna=True))

    dd = _normalize_data_dictionary(data_dictionary)
    dd_lookup = dd.set_index("variable", drop=False) if dd is not None else None

    results: list[dict] = []

    for col in num_list:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")
        for ms in missing_sentinels:
            x = x.mask(x == ms)

        mask = x.notna() & df[total_bal].notna() & df[bad_bal].notna()
        if int(mask.sum()) < min_non_missing:
            continue

        x = x.loc[mask]
        y = df.loc[mask, bad_flag]
        total_bal_sub = df.loc[mask, total_bal]
        bad_bal_sub = df.loc[mask, bad_bal]
        good_bal_sub = total_bal_sub - bad_bal_sub

        cleaned_name = _clean_variable_name(col)

        definition = None
        direction = direction_default
        if dd_lookup is not None and cleaned_name in dd_lookup.index:
            row = dd_lookup.loc[cleaned_name]
            definition = row.get("definition", None)
            d = row.get("direction", np.nan)
            if pd.notnull(d):
                try:
                    direction = int(float(d))
                except Exception:
                    direction = direction_default

        inferred_corr = np.nan
        direction_used = direction
        if direction_used == 0:
            inferred_corr = x.corr(y, method="spearman")
            if pd.isna(inferred_corr) or inferred_corr == 0:
                direction_used = 1
            else:
                direction_used = 1 if inferred_corr > 0 else -1

        # best tail selection
        if direction_used == 1:
            cutoff = float(x.quantile(best_pct))
            tail_mask = x <= cutoff
            tail_side = f"bottom_{int(best_pct*100)}%"
        else:
            cutoff = float(x.quantile(1 - best_pct))
            tail_mask = x >= cutoff
            tail_side = f"top_{int(best_pct*100)}%"

        tail_n = int(tail_mask.sum())
        if tail_n == 0:
            continue

        tail_bad_n = int(y.loc[tail_mask].sum())
        tail_bad_rate = tail_bad_n / tail_n
        tail_good_n = int(tail_n - tail_bad_n)

        tail_total_balance = float(total_bal_sub.loc[tail_mask].sum())
        tail_bad_balance = float(bad_bal_sub.loc[tail_mask].sum())
        tail_good_balance = float(good_bal_sub.loc[tail_mask].sum())

        good_balance_rate_in_tail = tail_good_balance / tail_total_balance if tail_total_balance > 0 else np.nan
        good_balance_share_of_total = tail_good_balance / total_good_balance_all if total_good_balance_all > 0 else np.nan

        results.append({
            "feature": col,
            "cleaned_name": cleaned_name,
            "definition": definition,
            "direction_used": direction_used,
            "tail_side": tail_side,
            "cutoff": cutoff,

            "tail_n": tail_n,
            "tail_good_n": tail_good_n,
            "tail_bad_n": tail_bad_n,
            "tail_bad_rate": tail_bad_rate,

            "tail_total_balance": tail_total_balance,
            "tail_good_balance": tail_good_balance,
            "tail_bad_balance": tail_bad_balance,

            "good_balance_rate_in_tail": good_balance_rate_in_tail,
            "good_balance_share_of_total": good_balance_share_of_total,

            "spearman_corr_if_inferred": inferred_corr,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # prioritize low bad rate, then more good balance
    out = out.sort_values(
        ["tail_bad_rate", "tail_good_balance", "good_balance_share_of_total"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# =============================================================================
# 2) Config dataframe construction (simple, row-by-row)
# =============================================================================

RULE_CFG_COLUMNS = [
    "feature",
    "min", "max",
    "search_min", "search_max", "search_step",
    "direction",  # 1 higher risky, -1 lower risky
]


def new_config_rule_df() -> pd.DataFrame:
    """Create an empty rule config dataframe with standard columns."""
    return pd.DataFrame(columns=RULE_CFG_COLUMNS)


# =============================================================================
# 3) Grid-search approval rule construction (1D / 2D / 3D) with min/max bounds
# =============================================================================

def _threshold_grid(search_min: float, search_max: float, search_step: float) -> np.ndarray:
    if any(pd.isna(v) for v in [search_min, search_max, search_step]):
        raise ValueError("search_min/search_max/search_step cannot be NaN")
    if search_step <= 0:
        raise ValueError("search_step must be > 0")
    return np.arange(search_min, search_max + search_step * 0.5, search_step)


def _bounded_approval_mask(
    df: pd.DataFrame,
    feature: str,
    *,
    direction: int,
    cutoff: float,
    min_val: Optional[float],
    max_val: Optional[float],
) -> np.ndarray:
    """
    Bounded approval rule mask.

    direction= 1 (high risky): approve if min < x <= cutoff  AND x < max
    direction=-1 (low risky): approve if cutoff <= x < max AND x > min
    """
    x = pd.to_numeric(df[feature], errors="coerce")

    if min_val is not None and not pd.isna(min_val):
        x = x.where(x > float(min_val))
    if max_val is not None and not pd.isna(max_val):
        x = x.where(x < float(max_val))

    if direction == 1:
        return (x.notna()) & (x <= cutoff)
    if direction == -1:
        return (x.notna()) & (x >= cutoff)
    raise ValueError(f"direction must be 1 or -1 (got {direction}) for feature={feature}")


def _metrics(df: pd.DataFrame, mask: np.ndarray, bad_flag: str) -> dict:
    """Metrics on the approved subset (the 'new baseline')."""
    approved_n = int(mask.sum())
    if approved_n == 0:
        return {
            "approved_n": 0,
            "approved_bad_n": 0,
            "approved_bad_rate": np.nan,
            "approved_good_n": 0,
            "coverage": 0.0,
        }

    y = pd.to_numeric(df.loc[mask, bad_flag], errors="coerce")
    y = y[y.isin([0, 1])].astype(int)

    approved_bad_n = int(y.sum())
    approved_good_n = int(len(y) - approved_bad_n)
    approved_bad_rate = approved_bad_n / len(y) if len(y) > 0 else np.nan
    coverage = approved_n / len(df)

    return {
        "approved_n": approved_n,
        "approved_bad_n": approved_bad_n,
        "approved_bad_rate": approved_bad_rate,
        "approved_good_n": approved_good_n,
        "coverage": coverage,
    }


def grid_search_1d_approval_rules(
    df: pd.DataFrame,
    config_rule: pd.DataFrame,
    *,
    bad_flag: str = "bad_flag",
    min_approved_n: int = 200,
    max_bad_rate: float = 0.02,
) -> pd.DataFrame:
    """Construct all possible 1D bounded approval rules over the grid defined in config_rule."""
    rows: list[dict] = []
    df_use = _clean_binary_target(df.copy(), bad_flag)

    for _, r in config_rule.iterrows():
        feature = r["feature"]
        if feature not in df_use.columns:
            continue

        direction = int(r["direction"])
        min_val = r.get("min", None)
        max_val = r.get("max", None)

        grid = _threshold_grid(float(r["search_min"]), float(r["search_max"]), float(r["search_step"]))
        for cutoff in grid:
            mask = _bounded_approval_mask(
                df_use, feature,
                direction=direction, cutoff=float(cutoff),
                min_val=min_val, max_val=max_val,
            )
            m = _metrics(df_use, mask, bad_flag)
            if m["approved_n"] >= min_approved_n and (pd.isna(m["approved_bad_rate"]) or m["approved_bad_rate"] <= max_bad_rate):
                op = "<=" if direction == 1 else ">="
                rule_txt = f"({min_val} < {feature} < {max_val}) AND ({feature} {op} {cutoff})"
                rows.append({
                    "rule_dim": 1,
                    "rule": rule_txt,
                    "feature_1": feature, "cutoff_1": float(cutoff), "dir_1": direction,
                    "min_1": min_val, "max_1": max_val,
                    **m
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["approved_bad_rate", "approved_n"], ascending=[True, False]).reset_index(drop=True)


def grid_search_2d_approval_rules(
    df: pd.DataFrame,
    config_rule: pd.DataFrame,
    *,
    bad_flag: str = "bad_flag",
    features: Optional[list[str]] = None,
    min_approved_n: int = 200,
    max_bad_rate: float = 0.02,
    top_k_1d_per_feature: int = 20,
) -> pd.DataFrame:
    """
    Construct 2D bounded approval rules:
    - Precompute 1D candidates per feature, keep top_k
    - Combine candidates for feature pairs
    """
    df_use = _clean_binary_target(df.copy(), bad_flag)
    cfg = config_rule.copy()
    if features is not None:
        cfg = cfg[cfg["feature"].isin(features)].copy()

    # Precompute 1D candidates
    cands: dict[str, list[tuple[float, int, object, object, dict]]] = {}
    for _, r in cfg.iterrows():
        feature = r["feature"]
        if feature not in df_use.columns:
            continue

        direction = int(r["direction"])
        min_val = r.get("min", None)
        max_val = r.get("max", None)

        grid = _threshold_grid(float(r["search_min"]), float(r["search_max"]), float(r["search_step"]))
        tmp = []
        for cutoff in grid:
            mask = _bounded_approval_mask(df_use, feature, direction=direction, cutoff=float(cutoff), min_val=min_val, max_val=max_val)
            m = _metrics(df_use, mask, bad_flag)
            if m["approved_n"] > 0:
                tmp.append((float(cutoff), direction, min_val, max_val, m))

        if tmp:
            tmp_sorted = sorted(tmp, key=lambda t: (np.inf if pd.isna(t[4]["approved_bad_rate"]) else t[4]["approved_bad_rate"], -t[4]["approved_n"]))
            cands[feature] = tmp_sorted[:top_k_1d_per_feature]

    rows: list[dict] = []
    for f1, f2 in itertools.combinations(list(cands.keys()), 2):
        for cut1, dir1, min1, max1, _ in cands[f1]:
            mask1 = _bounded_approval_mask(df_use, f1, direction=dir1, cutoff=cut1, min_val=min1, max_val=max1)
            for cut2, dir2, min2, max2, _ in cands[f2]:
                mask2 = _bounded_approval_mask(df_use, f2, direction=dir2, cutoff=cut2, min_val=min2, max_val=max2)
                mask = mask1 & mask2
                m = _metrics(df_use, mask, bad_flag)
                if m["approved_n"] >= min_approved_n and (pd.isna(m["approved_bad_rate"]) or m["approved_bad_rate"] <= max_bad_rate):
                    op1 = "<=" if dir1 == 1 else ">="
                    op2 = "<=" if dir2 == 1 else ">="
                    rule_txt = (
                        f"({min1} < {f1} < {max1}) AND ({f1} {op1} {cut1}) AND "
                        f"({min2} < {f2} < {max2}) AND ({f2} {op2} {cut2})"
                    )
                    rows.append({
                        "rule_dim": 2,
                        "rule": rule_txt,
                        "feature_1": f1, "cutoff_1": cut1, "dir_1": dir1, "min_1": min1, "max_1": max1,
                        "feature_2": f2, "cutoff_2": cut2, "dir_2": dir2, "min_2": min2, "max_2": max2,
                        **m
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["approved_bad_rate", "approved_n"], ascending=[True, False]).reset_index(drop=True)


def grid_search_3d_approval_rules(
    df: pd.DataFrame,
    config_rule: pd.DataFrame,
    *,
    bad_flag: str = "bad_flag",
    features: Optional[list[str]] = None,
    min_approved_n: int = 200,
    max_bad_rate: float = 0.01,
    top_k_1d_per_feature: int = 12,
) -> pd.DataFrame:
    """Construct 3D bounded approval rules (precompute top_k 1D candidates per feature)."""
    df_use = _clean_binary_target(df.copy(), bad_flag)
    cfg = config_rule.copy()
    if features is not None:
        cfg = cfg[cfg["feature"].isin(features)].copy()

    cands: dict[str, list[tuple[float, int, object, object, dict]]] = {}
    for _, r in cfg.iterrows():
        feature = r["feature"]
        if feature not in df_use.columns:
            continue

        direction = int(r["direction"])
        min_val = r.get("min", None)
        max_val = r.get("max", None)

        grid = _threshold_grid(float(r["search_min"]), float(r["search_max"]), float(r["search_step"]))
        tmp = []
        for cutoff in grid:
            mask = _bounded_approval_mask(df_use, feature, direction=direction, cutoff=float(cutoff), min_val=min_val, max_val=max_val)
            m = _metrics(df_use, mask, bad_flag)
            if m["approved_n"] > 0:
                tmp.append((float(cutoff), direction, min_val, max_val, m))

        if tmp:
            tmp_sorted = sorted(tmp, key=lambda t: (np.inf if pd.isna(t[4]["approved_bad_rate"]) else t[4]["approved_bad_rate"], -t[4]["approved_n"]))
            cands[feature] = tmp_sorted[:top_k_1d_per_feature]

    rows: list[dict] = []
    for f1, f2, f3 in itertools.combinations(list(cands.keys()), 3):
        for cut1, dir1, min1, max1, _ in cands[f1]:
            mask1 = _bounded_approval_mask(df_use, f1, direction=dir1, cutoff=cut1, min_val=min1, max_val=max1)
            for cut2, dir2, min2, max2, _ in cands[f2]:
                mask12 = mask1 & _bounded_approval_mask(df_use, f2, direction=dir2, cutoff=cut2, min_val=min2, max_val=max2)
                if mask12.sum() == 0:
                    continue
                for cut3, dir3, min3, max3, _ in cands[f3]:
                    mask = mask12 & _bounded_approval_mask(df_use, f3, direction=dir3, cutoff=cut3, min_val=min3, max_val=max3)
                    m = _metrics(df_use, mask, bad_flag)
                    if m["approved_n"] >= min_approved_n and (pd.isna(m["approved_bad_rate"]) or m["approved_bad_rate"] <= max_bad_rate):
                        op1 = "<=" if dir1 == 1 else ">="
                        op2 = "<=" if dir2 == 1 else ">="
                        op3 = "<=" if dir3 == 1 else ">="
                        rule_txt = (
                            f"({min1} < {f1} < {max1}) AND ({f1} {op1} {cut1}) AND "
                            f"({min2} < {f2} < {max2}) AND ({f2} {op2} {cut2}) AND "
                            f"({min3} < {f3} < {max3}) AND ({f3} {op3} {cut3})"
                        )
                        rows.append({
                            "rule_dim": 3,
                            "rule": rule_txt,
                            "feature_1": f1, "cutoff_1": cut1, "dir_1": dir1, "min_1": min1, "max_1": max1,
                            "feature_2": f2, "cutoff_2": cut2, "dir_2": dir2, "min_2": min2, "max_2": max2,
                            "feature_3": f3, "cutoff_3": cut3, "dir_3": dir3, "min_3": min3, "max_3": max3,
                            **m
                        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["approved_bad_rate", "approved_n"], ascending=[True, False]).reset_index(drop=True)


# =============================================================================
# 4) Final usage method (copy/paste)
# =============================================================================

USAGE_EXAMPLE = """
from approval_rules import (
    rank_features_by_best_tail_good_vol,
    new_config_rule_df,
    grid_search_1d_approval_rules,
    grid_search_2d_approval_rules,
    grid_search_3d_approval_rules,
)

# df: your dataset, must include 'bad_flag' and candidate feature columns
# num_list: candidate numerical features
# dd: optional data dictionary

# 1) Best-tail screening
best = rank_features_by_best_tail_good_vol(
    data=df,
    bad_flag="bad_flag",
    num_list=num_list,
    data_dictionary=dd,
    best_pct=0.05
)

top_feats = best["feature"].head(10).tolist()

# 2) Build config df row-by-row
config_rule = new_config_rule_df()
config_rule.loc[len(config_rule)] = ["RN",   300, 999, 650, 780, 10,  -1]
config_rule.loc[len(config_rule)] = ["UTIL", 0.0, 1.0, 0.10, 0.50, 0.05,  1]
config_rule.loc[len(config_rule)] = ["MOB",  0,   240, 6,   48,   6,   -1]

config_rule = config_rule[config_rule["feature"].isin(top_feats)].reset_index(drop=True)

# 3) Rule search
rules_1d = grid_search_1d_approval_rules(df, config_rule, bad_flag="bad_flag", min_approved_n=300, max_bad_rate=0.02)

focus = rules_1d["feature_1"].dropna().unique().tolist()[:10]
rules_2d = grid_search_2d_approval_rules(df, config_rule, bad_flag="bad_flag", features=focus,
                                        min_approved_n=300, max_bad_rate=0.015, top_k_1d_per_feature=20)

rules_3d = grid_search_3d_approval_rules(df, config_rule, bad_flag="bad_flag", features=focus,
                                        min_approved_n=300, max_bad_rate=0.01, top_k_1d_per_feature=10)

# 4) Inspect the best rule
rules_2d.head(10)
"""

