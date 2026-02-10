def rank_features_by_worst_tail_bad_vol(
    data: pd.DataFrame,
    bad_flag: str,
    num_list: list,
    data_dictionary: pd.DataFrame = None,
    worst_pct: float = 0.05,          # e.g. 0.05 for worst 5%
    min_non_missing: int = 200,
    missing_sentinels: tuple = (-9999,),
    direction_default: int = 0,       # 0 => infer if missing
) -> pd.DataFrame:
    """
    Rank numerical features by BAD VOLUME captured in the worst X% tail.
    """

    if not (0 < worst_pct < 1):
        raise ValueError("worst_pct must be between 0 and 1 (e.g., 0.05 for 5%).")

    df = data.copy()

    # --- target cleaning ---
    y = pd.to_numeric(df[bad_flag], errors="coerce")
    df = df.loc[y.isin([0, 1])].copy()
    df[bad_flag] = y.loc[df.index].astype(int)

    overall_bad_rate = df[bad_flag].mean()
    if pd.isna(overall_bad_rate):
        raise ValueError("Overall bad rate is NaN (bad_flag might be empty after cleaning).")

    # --- normalise data_dictionary columns (case-insensitive) ---
    dd = None
    if data_dictionary is not None:
        dd = data_dictionary.copy()
        dd.columns = [c.strip().lower() for c in dd.columns]

        # accept both "variable"/"definition"/"direction" in any case
        required = {"variable", "definition", "direction"}
        if not required.issubset(set(dd.columns)):
            raise ValueError(f"data_dictionary must include columns: {required} (case-insensitive). Got: {set(dd.columns)}")

        dd["variable"] = dd["variable"].astype(str).str.strip()

        # quick lookup table by variable
        dd_lookup = dd.set_index("variable", drop=False)

    def clean_variable_name(target_variable: str) -> str:
        s = str(target_variable)
        cleaned = s.split("_")[-1] if "_" in s else s
        cleaned = cleaned.rstrip(" -") if " -" in cleaned else cleaned
        return cleaned

    results = []

    for col in num_list:
        if col not in df.columns:
            continue

        s_raw = pd.to_numeric(df[col], errors="coerce")

        # treat sentinels as missing
        for ms in missing_sentinels:
            s_raw = s_raw.mask(s_raw == ms)

        mask = s_raw.notna()
        n_non_missing = int(mask.sum())
        if n_non_missing < min_non_missing:
            continue

        x = s_raw.loc[mask]
        y_sub = df.loc[mask, bad_flag]

        cleaned_name = clean_variable_name(col)

        # --- get direction & definition from dictionary if possible ---
        definition = None
        direction = direction_default

        if dd is not None and cleaned_name in dd_lookup.index:
            row = dd_lookup.loc[cleaned_name]
            definition = row.get("definition", None)
            d = row.get("direction", np.nan)
            if pd.notnull(d):
                try:
                    direction = int(float(d))
                except Exception:
                    direction = direction_default

        # direction meaning:
        #  1 => high values are worse (worst tail = top X%)
        # -1 => low values are worse (worst tail = bottom X%)
        #  0/unknown => infer via Spearman correlation with target
        inferred_corr = np.nan
        direction_used = direction

        if direction_used == 0:
            inferred_corr = x.corr(y_sub, method="spearman")
            if pd.isna(inferred_corr) or inferred_corr == 0:
                direction_used = 1  # fallback
            else:
                direction_used = 1 if inferred_corr > 0 else -1

        # --- compute worst-tail cutoff + bad rate ---
        if direction_used == 1:
            cutoff = float(x.quantile(1 - worst_pct))
            tail_mask = x >= cutoff
            tail_side = f"top_{int(worst_pct*100)}%"
        else:
            cutoff = float(x.quantile(worst_pct))
            tail_mask = x <= cutoff
            tail_side = f"bottom_{int(worst_pct*100)}%"

        tail_n = int(tail_mask.sum())
        if tail_n == 0:
            continue

        tail_bad_n = int(y_sub.loc[tail_mask].sum())
        tail_bad_rate = tail_bad_n / tail_n

        results.append({
            "feature": col,                         # raw dataset column name
            "cleaned_name": cleaned_name,          # cleaned name used for dictionary matching
            "definition": definition,              # from data_dictionary['definition']
            "direction_used": direction_used,      # +1 high_worse / -1 low_worse
            "tail_side": tail_side,
            "cutoff": cutoff,
            "tail_n": tail_n,                      # volume
            "tail_bad_n": tail_bad_n,              # bad volume
            "tail_bad_rate": tail_bad_rate,        # bad rate in worst tail
            "overall_bad_rate": overall_bad_rate,
            "spearman_corr_if_inferred": inferred_corr
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values(["tail_bad_rate", "tail_n"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out




def rank_features_by_worst_tail_bad_bal(
    data: pd.DataFrame,
    bad_flag: str,
    total_bal: str,          
    bad_bal: str,
    num_list: list,
    data_dictionary: pd.DataFrame = None,
    worst_pct: float = 0.05,
    min_non_missing: int = 200,
    missing_sentinels: tuple = (-9999,),
    direction_default: int = 0,
):
    """
    Rank numerical features by BAD BALANCE captured in the worst X% tail.

    Uses:
      - total_bal: total exposure/balance
      - bad_bal: bad exposure/balance
    """

    df = data.copy()

    # --- clean target ---
    y = pd.to_numeric(df[bad_flag], errors="coerce")
    df = df.loc[y.isin([0, 1])].copy()
    df[bad_flag] = y.loc[df.index].astype(int)

    # --- balances ---
    df[total_bal] = pd.to_numeric(df[total_bal], errors="coerce")
    df[bad_bal] = pd.to_numeric(df[bad_bal], errors="coerce")

    # total bad balance (baseline reference)
    total_bad_balance_all = df[bad_bal].sum(skipna=True)

    # --- prepare data dictionary ---
    dd = None
    dd_lookup = None
    if data_dictionary is not None:
        dd = data_dictionary.copy()
        dd.columns = [c.strip().lower() for c in dd.columns]

        required = {"variable", "definition", "direction"}
        if not required.issubset(set(dd.columns)):
            raise ValueError(f"data_dictionary must include columns: {required}")

        dd["variable"] = dd["variable"].astype(str).str.strip()
        dd_lookup = dd.set_index("variable", drop=False)

    def clean_variable_name(target_variable: str) -> str:
        s = str(target_variable)
        cleaned = s.split("_")[-1] if "_" in s else s
        cleaned = cleaned.rstrip(" -") if " -" in cleaned else cleaned
        return cleaned

    results = []

    for col in num_list:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")

        # treat sentinels as missing
        for ms in missing_sentinels:
            x = x.mask(x == ms)

        # need x + balances
        mask = x.notna() & df[total_bal].notna() & df[bad_bal].notna()
        n_non_missing = int(mask.sum())
        if n_non_missing < min_non_missing:
            continue

        x = x.loc[mask]
        y_sub = df.loc[mask, bad_flag]
        total_bal_sub = df.loc[mask, total_bal]
        bad_bal_sub = df.loc[mask, bad_bal]

        cleaned_name = clean_variable_name(col)

        # --- dictionary info ---
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

        # infer direction if missing
        inferred_corr = np.nan
        direction_used = direction

        if direction_used == 0:
            inferred_corr = x.corr(y_sub, method="spearman")
            if pd.isna(inferred_corr) or inferred_corr == 0:
                direction_used = 1
            else:
                direction_used = 1 if inferred_corr > 0 else -1

        # --- determine worst tail ---
        if direction_used == 1:
            cutoff = float(x.quantile(1 - worst_pct))
            tail_mask = x >= cutoff
            tail_side = f"top_{int(worst_pct*100)}%"
        else:
            cutoff = float(x.quantile(worst_pct))
            tail_mask = x <= cutoff
            tail_side = f"bottom_{int(worst_pct*100)}%"

        tail_n = int(tail_mask.sum())
        if tail_n == 0:
            continue

        # --- volume metrics ---
        tail_bad_n = int(y_sub.loc[tail_mask].sum())
        tail_bad_rate = tail_bad_n / tail_n

        # --- balance metrics ---
        tail_total_balance = float(total_bal_sub.loc[tail_mask].sum())
        tail_bad_balance = float(bad_bal_sub.loc[tail_mask].sum())

        bad_balance_rate_in_tail = (
            tail_bad_balance / tail_total_balance if tail_total_balance > 0 else np.nan
        )

        bad_balance_share_of_total = (
            tail_bad_balance / total_bad_balance_all if total_bad_balance_all > 0 else np.nan
        )

        results.append({
            "feature": col,
            "cleaned_name": cleaned_name,
            "definition": definition,
            "direction_used": direction_used,
            "tail_side": tail_side,
            "cutoff": cutoff,

            "tail_n": tail_n,
            "tail_bad_n": tail_bad_n,
            "tail_bad_rate": tail_bad_rate,

            "tail_total_balance": tail_total_balance,
            "tail_bad_balance": tail_bad_balance,
            "bad_balance_rate_in_tail": bad_balance_rate_in_tail,
            "bad_balance_share_of_total": bad_balance_share_of_total,

            "spearman_corr_if_inferred": inferred_corr,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Rank by BAD BALANCE captured (and share)
    out = out.sort_values(
        ["tail_bad_balance", "bad_balance_share_of_total"],
        ascending=[False, False]
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out