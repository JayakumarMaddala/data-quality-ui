APP_VERSION = "2024-11-24-01"   # Version label for this analyzer module. Change when you update the logic.

# ---------- Imports ----------
import os
import io
import json
import math
import itertools
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import yaml

# -------------------------
# I/O Helpers (robust)
# -------------------------
def read_table(fileobj) -> Dict[str, pd.DataFrame] or pd.DataFrame:
    """
    Read a CSV or Excel uploaded file.
    Accepts:
      - path string
      - file-like object with .read() and .name
      - bytes
    Returns:
      - For Excel with multiple sheets: dict sheet_name -> DataFrame
      - For CSV or single-sheet Excel: DataFrame
    """
    # If string: treat as filesystem path
    if isinstance(fileobj, str):
        path = fileobj
        name = os.path.basename(path).lower()

        if name.endswith(".csv"):
            return pd.read_csv(path)

        if name.endswith((".xls", ".xlsx")):
            sheets = pd.read_excel(path, sheet_name=None)
            if len(sheets) == 1:
                return next(iter(sheets.values()))
            return sheets

        # Fallback: try CSV then Excel
        try:
            return pd.read_csv(path)
        except Exception:
            sheets = pd.read_excel(path, sheet_name=None)
            if len(sheets) == 1:
                return next(iter(sheets.values()))
            return sheets

    # If raw bytes
    if isinstance(fileobj, (bytes, bytearray)):
        b = bytes(fileobj)

        # Try Excel first
        try:
            x = pd.read_excel(io.BytesIO(b), sheet_name=None)
            if len(x) == 1:
                return next(iter(x.values()))
            return x
        except Exception:
            # Then CSV via decoded text
            try:
                return pd.read_csv(io.StringIO(b.decode("utf-8", errors="replace")))
            except Exception:
                # Last fallback: treat bytes directly
                return pd.read_csv(io.BytesIO(b))

    # File-like object (e.g., Streamlit upload)
    try:
        name = getattr(fileobj, "name", None)
    except Exception:
        name = None

    try:
        if name and isinstance(name, str) and name.lower().endswith(".csv"):
            try:
                fileobj.seek(0)
            except Exception:
                pass
            return pd.read_csv(fileobj)

        if name and isinstance(name, str) and name.lower().endswith((".xls", ".xlsx")):
            try:
                fileobj.seek(0)
            except Exception:
                pass
            sheets = pd.read_excel(fileobj, sheet_name=None)
            if len(sheets) == 1:
                return next(iter(sheets.values()))
            return sheets
    except Exception:
        # Fall through to generic logic
        pass

    # Generic Excel then CSV
    try:
        try:
            fileobj.seek(0)
        except Exception:
            pass
        sheets = pd.read_excel(fileobj, sheet_name=None)
        if len(sheets) == 1:
            return next(iter(sheets.values()))
        return sheets
    except Exception:
        try:
            try:
                fileobj.seek(0)
            except Exception:
                pass
            return pd.read_csv(fileobj)
        except Exception as e:
            raise RuntimeError(f"Unable to read uploaded file: {e}")


def save_report_json(report: Dict[str, Any], path: str):
    """Save the final report dictionary as a JSON file at the given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def load_business_key_map(path: Optional[str]) -> Dict[str, List[List[str]]]:
    """
    Load a business key config file (YAML or JSON) and return it as a dict.
    If path is missing or invalid, return empty dict.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[WARNING] business key config not found at: {path}")
        return {}
    try:
        if path.lower().endswith((".yml", ".yaml")):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception as e:
        print(f"[WARNING] Failed to load business key map: {e}")
        return {}

# -------------------------
# Basic Stats / PK inference / FK inference
# -------------------------
def basic_column_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute basic statistics for every column in a DataFrame."""
    stats: Dict[str, Dict[str, Any]] = {}
    n = len(df)

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        unique = non_null.nunique(dropna=True)
        pct_unique = unique / n if n > 0 else 0
        pct_null = (n - len(non_null)) / n if n > 0 else 0
        dtype = str(s.dtype)

        stats[col] = {
            "n": n,
            "non_null_count": int(len(non_null)),
            "null_count": int(n - len(non_null)),
            "pct_null": float(pct_null),
            "unique_count": int(unique),
            "pct_unique": float(pct_unique),
            "dtype": dtype,
            "sample_values": list(non_null.head(5).astype(str).unique()),
        }

    return stats


def score_pk_candidate(col: str, stats_col: Dict[str, Any]) -> float:
    """Assign a numeric score to a column to judge if it is a good primary key."""
    score = 0.0
    score += stats_col["pct_unique"] * 60
    score += (1 - stats_col["pct_null"]) * 20

    if "int" in stats_col["dtype"] or "float" in stats_col["dtype"]:
        score += 5
    if "object" in stats_col["dtype"] or "str" in stats_col["dtype"]:
        score += 3

    lname = col.lower()

    if lname == "id" or lname.endswith("id") or lname.endswith("_id"):
        score += 30

    if "code" in lname or "key" in lname or "pk" in lname or "name" in lname:
        score += 2

    return score


def find_single_pk(df: pd.DataFrame, min_pct_unique: float = 0.99) -> Optional[str]:
    """Try to find a single-column primary key."""
    if df is None or len(df) == 0:
        return None

    stats = basic_column_stats(df)
    scored: List[Tuple[str, float]] = []

    for col, st in stats.items():
        if st["non_null_count"] > 0 and st["unique_count"] >= st["non_null_count"]:
            scored.append((col, score_pk_candidate(col, st)))
        else:
            if st["pct_unique"] >= min_pct_unique and st["pct_null"] < 0.05:
                scored.append((col, score_pk_candidate(col, st)))

    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def find_composite_pk(df: pd.DataFrame, max_comb: int = 2) -> Optional[List[str]]:
    """Try to find a multi-column (composite) primary key."""
    if df is None or len(df) == 0:
        return None

    stats = basic_column_stats(df)
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["pct_unique"], reverse=True)
    top_cols = [c for c, s in ranked[:10]]

    for r in range(2, max_comb + 1):
        for combo in itertools.combinations(top_cols, r):
            combined = df[list(combo)].astype(str).fillna("__NULL__").agg("||".join, axis=1)
            nonnull_mask = ~df[list(combo)].isnull().any(axis=1)
            if combined[nonnull_mask].nunique(dropna=True) >= nonnull_mask.sum():
                return list(combo)

    return None


def infer_pk(df: pd.DataFrame) -> Dict[str, Any]:
    """Infer primary key information for a table."""
    if df is None:
        return {"single": None, "composite": None, "stats": {}}

    single = find_single_pk(df, min_pct_unique=0.5)  # relaxed uniqueness
    composite = None if single else find_composite_pk(df, max_comb=2)
    stats = basic_column_stats(df)

    return {"single": single, "composite": composite, "stats": stats}


def find_fk_candidates(
    tables: Dict[str, pd.DataFrame],
    pks: Dict[str, Dict[str, Any]],
    subset_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Infer foreign-key relationships between tables.

    Phase 1: value-based (existing behavior)
      - Look for columns whose values overlap a parent table's PK values.

    Phase 2: name-based hints (NEW)
      - If no value-based match was found for a child Id-like column,
        try to guess the parent by column/table naming conventions
        (e.g. Subscription.ContractId -> Contract.Id).
      - These hinted relationships may have matched_values_count == 0,
        but they still allow orphan detection to flag all non-matching rows.
    """

    def _norm_name(s: str) -> str:
        """Normalize names: lowercase, keep only letters/numbers."""
        return "".join(ch.lower() for ch in str(s) if ch.isalnum())

    rels: List[Dict[str, Any]] = []

    # ------------------ Precompute PK value sets ------------------
    pk_values: Dict[str, Dict[str, Any]] = {}

    for tname, pinfo in pks.items():
        if pinfo is None:
            pk_values[tname] = {}
            continue

        if pinfo.get("single"):
            col = pinfo["single"]
            if tname not in tables or col not in tables[tname].columns:
                pk_values[tname] = {}
                continue
            vals = set(tables[tname][col].dropna().astype(str).unique())
            pk_values[tname] = {"cols": [col], "values": vals}

        elif pinfo.get("composite"):
            cols = pinfo["composite"]
            if tname not in tables or not all(c in tables[tname].columns for c in cols):
                pk_values[tname] = {}
                continue
            combined = (
                tables[tname][cols]
                .astype(str)
                .fillna("__NULL__")
                .agg("||".join, axis=1)
            )
            pk_values[tname] = {"cols": cols, "values": set(combined.unique())}
        else:
            pk_values[tname] = {}

    # ------------------ Phase 1: value-based FK inference ------------------
    for child_name, df in tables.items():
        if df is None or len(df) == 0:
            continue

        for col in df.columns:
            col_vals = set(df[col].dropna().astype(str).unique())
            if not col_vals:
                continue

            for parent_name, pkinfo in pk_values.items():
                if parent_name == child_name or not pkinfo.get("values"):
                    continue

                parent_vals = pkinfo["values"]
                intersection = col_vals.intersection(parent_vals)
                if not intersection:
                    continue

                overlap_prop = len(intersection) / len(col_vals)
                parent_coverage = len(intersection) / len(parent_vals) if parent_vals else 0

                if overlap_prop >= subset_threshold and len(intersection) >= 1:
                    try:
                        child_nonnull = df[[col]].dropna().astype(str)
                        parent_map_counts: Dict[str, int] = {}
                        for v in intersection:
                            parent_map_counts[v] = int(
                                (child_nonnull[col].astype(str) == v).sum()
                            )
                        avg_children = (
                            sum(parent_map_counts.values()) / len(parent_map_counts)
                            if parent_map_counts
                            else 0
                        )
                        child_distinct = len(col_vals)
                        parent_distinct = len(parent_vals)

                        if avg_children > 1.5:
                            cardinality = "One-to-Many"
                        else:
                            if (
                                child_distinct <= len(intersection)
                                and len(intersection) >= parent_distinct * 0.9
                            ):
                                cardinality = "Many-to-One"
                            else:
                                cardinality = "One-to-Many" if avg_children >= 1 else "Many-to-Many"
                    except Exception:
                        cardinality = "Unknown"

                    rels.append(
                        {
                            "from": child_name,
                            "fk": col,
                            "to": parent_name,
                            "to_pk_candidate": pkinfo.get("cols", []),
                            "matched_values_count": len(intersection),
                            "overlap_prop": round(overlap_prop, 3),
                            "parent_coverage": round(parent_coverage, 3),
                            "cardinality": cardinality,
                            "match_hint": "values",
                        }
                    )

    # Record which relations already exist
    existing_rel_keys = {(r["from"], r["fk"], r["to"]) for r in rels}

    # ------------------ Phase 2: name-based hints ------------------
    table_norm = {t: _norm_name(t) for t in tables.keys()}

    for child_name, df in tables.items():
        if df is None or len(df) == 0:
            continue

        child_norm = table_norm.get(child_name, _norm_name(child_name))

        for col in df.columns:
            if not col.lower().endswith("id"):
                continue

            col_norm = _norm_name(col)

            for parent_name, pinfo in pks.items():
                if parent_name == child_name:
                    continue
                if not pinfo or not pinfo.get("single"):
                    continue

                pkcol = pinfo["single"]
                parent_norm = table_norm.get(parent_name, _norm_name(parent_name))
                pk_norm = _norm_name(pkcol)

                if (child_name, col, parent_name) in existing_rel_keys:
                    continue

                rule1 = col_norm == parent_norm + "id"
                rule2 = col_norm.endswith(parent_norm + "id")
                rule3 = col_norm == pk_norm
                rule4 = parent_norm in col_norm and col_norm.endswith("id")

                if not (rule1 or rule2 or rule3 or rule4):
                    continue

                rels.append(
                    {
                        "from": child_name,
                        "fk": col,
                        "to": parent_name,
                        "to_pk_candidate": [pkcol],
                        "matched_values_count": 0,
                        "overlap_prop": 0.0,
                        "parent_coverage": 0.0,
                        "cardinality": "Unknown",
                        "match_hint": "name_only",
                    }
                )
                existing_rel_keys.add((child_name, col, parent_name))

    # Sort: value-based first, then name-only
    rels = sorted(
        rels,
        key=lambda x: (
            0 if x.get("match_hint") == "values" else 1,
            -x.get("overlap_prop", 0.0),
        ),
    )

    return rels

# -------------------------
# Orphan detection (NEW)
# -------------------------
def detect_orphans(
    tables: Dict[str, pd.DataFrame],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Orphan detection in two phases:

    1) Relationship-based:
       For each inferred FK relationship, find child rows whose FK value
       does NOT exist in the parent PK column.

    2) Global ID-based:
       For every Id-like column (col name ends with 'Id'), if there is
       no explicit relationship using that column, check its values
       against *all* Id/Id-like columns in the dataset.
       Values that are not found anywhere are reported as "global" orphans.
    """
    orphans: List[Dict[str, Any]] = []

    # ---------- 1) Relationship-based orphans ----------
    for r in relationships:
        child = r["from"]
        parent = r["to"]
        fk = r["fk"]
        pk_cols = r.get("to_pk_candidate", [])

        if isinstance(pk_cols, list) and len(pk_cols) == 1:
            pkcol = pk_cols[0]
            if parent not in tables or child not in tables:
                continue

            parent_vals = set(tables[parent][pkcol].dropna().astype(str).unique())
            child_df = tables[child]

            missing_mask = ~child_df[fk].astype(str).isin(parent_vals)
            missing_mask = missing_mask & child_df[fk].notna()

            count_missing = int(missing_mask.sum())
            if count_missing > 0:
                examples = (
                    child_df.loc[missing_mask, [fk]]
                    .head(10)
                    .to_dict(orient="records")
                )
                orphans.append(
                    {
                        "from": child,
                        "to": parent,
                        "fk": fk,
                        "count": count_missing,
                        "examples": examples,
                        "source": "relationship",
                    }
                )

    handled_pairs = {(o["from"], o["fk"]) for o in orphans}

    # ---------- 2) Global ID-based orphans ----------
    # 2a) Build global parent ID pools per Id-like column name
    parent_value_sets: Dict[str, set] = {}

    for tname, df in tables.items():
        if df is None or not isinstance(df, pd.DataFrame):
            continue

        for col in df.columns:
            if col.lower().endswith("id"):
                vals = df[col].dropna().astype(str).unique()
                parent_value_sets.setdefault(col, set()).update(vals)

    # 2b) For each Id-like column not covered by a relationship
    for child_name, df in tables.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        for col in df.columns:
            if not col.lower().endswith("id"):
                continue

            if (child_name, col) in handled_pairs:
                continue

            series = df[col].dropna().astype(str)
            if series.empty:
                continue

            known_ids = set()
            if col in parent_value_sets:
                known_ids |= parent_value_sets[col]
            if "Id" in parent_value_sets and col != "Id":
                known_ids |= parent_value_sets["Id"]

            if not known_ids:
                continue

            missing_mask = ~series.isin(known_ids)
            if not missing_mask.any():
                continue

            missing_indices = series.index[missing_mask]
            count_missing = int(missing_mask.sum())

            examples = (
                df.loc[missing_indices, [col]]
                .head(10)
                .to_dict(orient="records")
            )

            orphans.append(
                {
                    "from": child_name,
                    "to": None,
                    "fk": col,
                    "count": count_missing,
                    "examples": examples,
                    "source": "global_id",
                }
            )

    return orphans

# -------------------------
# Logical & near-duplicate detection
# -------------------------
def detect_logical_duplicates(df: pd.DataFrame, pk_col: str) -> Dict[str, Any]:
    """
    Detect "logical duplicates": rows with the same non-PK fields but different PKs.
    Returns:
      - groups: which signatures have multiple PKs
      - mapping: duplicate_pk -> canonical_pk
    """
    result = {"groups": {}, "mapping": {}}

    if df is None or pk_col is None or pk_col not in df.columns:
        return result

    non_pk_cols = [c for c in df.columns if c != pk_col]
    if not non_pk_cols:
        return result

    sig_series = df[non_pk_cols].astype(str).fillna("__NULL__").agg("||".join, axis=1)
    grouped: Dict[str, Dict[str, Any]] = {}

    for sig, pk_val, row in zip(sig_series, df[pk_col].astype(str), df.to_dict(orient="records")):
        grouped.setdefault(sig, {"pk_values": [], "rows": []})
        grouped[sig]["pk_values"].append(str(pk_val))
        grouped[sig]["rows"].append(row)

    mapping: Dict[str, str] = {}
    groups_out: Dict[str, Any] = {}

    for sig, info in grouped.items():
        if len(set(info["pk_values"])) > 1:
            pk_values = info["pk_values"]
            canonical = pk_values[0]
            duplicates = [p for p in pk_values if p != canonical]

            for d in duplicates:
                mapping[d] = canonical

            groups_out[sig] = {
                "canonical": canonical,
                "pk_values": pk_values,
                "rows": info["rows"],
            }

    result["groups"] = groups_out
    result["mapping"] = mapping
    return result


def detect_near_duplicates(df: pd.DataFrame, pk_col: str, numeric_tolerance: float = 0.05) -> List[Dict[str, Any]]:
    """
    Detect "near duplicates":
    - All non-numeric columns are exactly the same.
    - Numeric columns are close within a given tolerance.
    """
    results: List[Dict[str, Any]] = []

    if df is None or pk_col is None or pk_col not in df.columns:
        return results

    cols = list(df.columns)
    non_pk_cols = [c for c in cols if c != pk_col]
    rows = df.to_dict(orient="records")

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r1 = rows[i]
            r2 = rows[j]
            pk1 = str(r1[pk_col])
            pk2 = str(r2[pk_col])

            nonnumeric_equal = True
            numeric_diffs: Dict[str, float] = {}

            for c in non_pk_cols:
                v1 = r1.get(c)
                v2 = r2.get(c)

                if v1 is None and v2 is None:
                    continue
                try:
                    n1 = float(v1)
                    n2 = float(v2)
                    if n1 == 0 and n2 == 0:
                        diff_frac = 0.0
                    else:
                        denom = max(abs(n1), abs(n2), 1e-9)
                        diff_frac = abs(n1 - n2) / denom
                    numeric_diffs[c] = diff_frac
                except Exception:
                    s1 = str(v1).strip().lower() if v1 is not None else ""
                    s2 = str(v2).strip().lower() if v2 is not None else ""
                    if s1 != s2:
                        nonnumeric_equal = False
                        break

            if not nonnumeric_equal:
                continue

            if numeric_diffs:
                max_diff = max(numeric_diffs.values()) if numeric_diffs else 0.0
                if 0 < max_diff <= numeric_tolerance:
                    results.append(
                        {
                            "pk_pair": (pk1, pk2),
                            "numeric_diffs": numeric_diffs,
                            "max_diff": max_diff,
                            "rows": [r1, r2],
                        }
                    )

    return results

# -------------------------
# Additional validation helpers
# -------------------------
def check_pk_uniqueness(df: pd.DataFrame, pk_col: str) -> Dict[str, Any]:
    """Check if the given PK column has duplicate values."""
    out = {"pk_col": pk_col, "dup_count_total": 0, "dup_values": {}}
    if df is None or pk_col is None or pk_col not in df.columns:
        return out

    dup_mask_any = df.duplicated(subset=[pk_col], keep=False)
    if dup_mask_any.sum() == 0:
        return out

    dup_df = df.loc[dup_mask_any, :].copy()
    grouped = dup_df.groupby(pk_col)

    total_dups = 0
    for key_val, group in grouped:
        cnt = len(group)
        if cnt > 1:
            total_dups += cnt - 1
            out["dup_values"][str(key_val)] = {
                "count": int(cnt),
                "examples": group.head(5).to_dict(orient="records"),
            }

    out["dup_count_total"] = int(total_dups)
    return out


def check_full_row_uniqueness(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for full-row duplicates (entire row identical)."""
    res = {"total_groups": 0, "groups": {}}
    if df is None or df.shape[0] == 0:
        return res

    sig = df.astype(str).fillna("__NULL__").agg("||".join, axis=1)
    df_sig = df.assign(__sig__=sig)
    grouped = df_sig.groupby("__sig__")

    for s, g in grouped:
        if len(g) > 1:
            res["total_groups"] += 1
            res["groups"][s] = {
                "count": len(g),
                "examples": g.drop(columns="__sig__").head(5).to_dict(orient="records"),
            }

    return res


def check_business_key_uniqueness(df: pd.DataFrame, business_keys: List[List[str]]) -> Dict[str, Any]:
    """Check duplicates for each business key (one or more columns)."""
    res = {"checked_keys": []}
    if df is None or df.shape[0] == 0:
        return res

    for bk in business_keys:
        if not all(c in df.columns for c in bk):
            res["checked_keys"].append({"key": bk, "status": "missing_columns"})
            continue

        dup_mask = df.duplicated(subset=bk, keep=False)
        dup_count = int(dup_mask.sum())
        details = None

        if dup_count > 0:
            grouped = df.loc[dup_mask].groupby(bk)
            vals: Dict[str, Any] = {}
            for k, g in grouped:
                if len(g) > 1:
                    if isinstance(k, tuple):
                        valname = "||".join([str(x) for x in k])
                    else:
                        valname = str(k)
                    vals[valname] = {
                        "count": len(g),
                        "examples": g.head(5).to_dict(orient="records"),
                    }
            details = {"total_duplicates": dup_count, "values": vals}

        res["checked_keys"].append({"key": bk, "dup_count": dup_count, "details": details})
    return res


def check_referential_integrity(
    tables: Dict[str, pd.DataFrame],
    relationships: List[Dict[str, Any]],
    pks: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Expanded referential integrity check supporting single and composite PKs.
    """
    out: List[Dict[str, Any]] = []

    for r in relationships:
        child = r["from"]
        parent = r["to"]
        fk_col = r["fk"]
        pk_cols = r.get("to_pk_candidate", [])

        if child not in tables or parent not in tables:
            continue

        child_df = tables[child]
        parent_df = tables[parent]

        if isinstance(pk_cols, list) and len(pk_cols) == 1:
            pkc = pk_cols[0]
            parent_vals = set(parent_df[pkc].dropna().astype(str).unique())
            missing_mask = ~child_df[fk_col].astype(str).isin(parent_vals)
            missing_mask = missing_mask & child_df[fk_col].notna()
            missing_count = int(missing_mask.sum())
            if missing_count > 0:
                out.append(
                    {
                        "from": child,
                        "to": parent,
                        "fk": fk_col,
                        "pk_cols": [pkc],
                        "count": missing_count,
                        "examples": child_df.loc[missing_mask].head(10).to_dict(orient="records"),
                    }
                )
        else:
            pk_info = pks.get(parent, {})
            comp_cols = pk_info.get("composite") or pk_cols
            if not comp_cols:
                continue

            if all(c in child_df.columns for c in comp_cols):
                combined_parent = parent_df[comp_cols].astype(str).fillna("__NULL__").agg("||".join, axis=1)
                parent_set = set(combined_parent.unique())
                combined_child = child_df[comp_cols].astype(str).fillna("__NULL__").agg("||".join, axis=1)
                missing_mask = ~combined_child.isin(parent_set) & combined_child.notna()
                missing_count = int(missing_mask.sum())
                if missing_count > 0:
                    out.append(
                        {
                            "from": child,
                            "to": parent,
                            "fk": comp_cols,
                            "pk_cols": comp_cols,
                            "count": missing_count,
                            "examples": child_df.loc[missing_mask, comp_cols].head(10).to_dict(orient="records"),
                        }
                    )

    return out


def check_fk_consistency_for_merged_parents(
    tables: Dict[str, pd.DataFrame],
    relationships: List[Dict[str, Any]],
    parent_merge_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    After parent logical merges (duplicate parents mapped to canonical IDs),
    check child tables to see where FKs still point to old duplicate parent IDs.
    """
    out = {"conflicts": [], "summary": {"total_conflicts": 0, "total_rows_affected": 0}}
    if not parent_merge_map:
        return out

    for rel in relationships:
        child = rel["from"]
        fk = rel["fk"]

        if child not in tables:
            continue
        child_df = tables[child]
        if fk not in child_df.columns:
            continue

        seen: Dict[str, List[Dict[str, Any]]] = {}

        for dup_val, canon in parent_merge_map.items():
            mask = child_df[fk].astype(str) == str(dup_val)
            if mask.any():
                seen.setdefault(canon, []).append(
                    {
                        "dup_val": dup_val,
                        "count": int(mask.sum()),
                        "examples": child_df.loc[mask].head(5).to_dict(orient="records"),
                    }
                )

        for canon, entries in seen.items():
            mask_canon = child_df[fk].astype(str) == str(canon)
            canonical_count = int(mask_canon.sum())
            rows_affected = sum(e["count"] for e in entries)

            out["conflicts"].append(
                {
                    "child_table": child,
                    "fk_column": fk,
                    "canonical": canon,
                    "duplicate_values": entries,
                    "canonical_count": canonical_count,
                    "rows_affected": rows_affected,
                    "suggested_updates": [
                        f"UPDATE {child} SET {fk} = '{canon}' WHERE {fk} IN ({', '.join([repr(e['dup_val']) for e in entries])});"
                    ],
                }
            )
            out["summary"]["total_conflicts"] += 1
            out["summary"]["total_rows_affected"] += rows_affected

    return out


def check_one_to_one_integrity(
    tables: Dict[str, pd.DataFrame],
    relationships: List[Dict[str, Any]],
    enforce_unique_fk_threshold: float = 0.95,
) -> List[Dict[str, Any]]:
    """
    For relationships where the child FK looks almost unique (> threshold),
    treat it as intended 1:1 and check for violations.
    """
    issues: List[Dict[str, Any]] = []

    for rel in relationships:
        child = rel["from"]
        fk = rel["fk"]
        parent = rel["to"]

        if child not in tables:
            continue
        child_df = tables[child]
        if fk not in child_df.columns:
            continue

        nonnull = child_df[fk].dropna().astype(str)
        if len(nonnull) == 0:
            continue

        unique_count = nonnull.nunique(dropna=True)
        pct_unique = unique_count / len(nonnull)

        if pct_unique >= enforce_unique_fk_threshold:
            counts = nonnull.value_counts()
            viol = counts[counts > 1]

            if not viol.empty:
                violations = []
                for val, cnt in viol.head(20).items():
                    mask = child_df[fk].astype(str) == val
                    violations.append(
                        {
                            "parent_value": val,
                            "ref_count": int(cnt),
                            "examples": child_df.loc[mask].head(5).to_dict(orient="records"),
                        }
                    )
                issues.append(
                    {
                        "child_table": child,
                        "fk_column": fk,
                        "parent_table": parent,
                        "reason": f"FK appears intended as 1:1 (pct_unique={pct_unique:.2f}) but found parent ids referenced multiple times",
                        "violations": violations,
                        "stats": {"pct_unique": pct_unique, "rows_examined": len(nonnull)},
                    }
                )

    return issues

# -------------------------
# Main analysis & reporting
# -------------------------
def generate_analysis_report(
    raw_tables: Dict[str, Any],
    business_key_map: Optional[Dict[str, List[List[str]]]] = None,
    output_folder: str = "analysis_output",
) -> Dict[str, Any]:
    """
    Main entry point for analyzer.
    """
    os.makedirs(output_folder, exist_ok=True)

    # ---- Step 1: flatten multi-sheet inputs into simple table dict ----
    tables: Dict[str, pd.DataFrame] = {}
    for name, obj in raw_tables.items():
        if isinstance(obj, dict):
            for sheet, df in obj.items():
                tname = f"{name}__{sheet}"
                tables[tname] = df.reset_index(drop=True).copy()
        else:
            tables[name] = obj.reset_index(drop=True).copy()

    # Validation summary
    validation = {
        "uploaded_count": len(tables),
        "validated_count": 0,
        "invalid_count": 0,
        "details": {},
        "files_analyzed": [],
        "files_skipped": [],
    }

    pks: Dict[str, Dict[str, Any]] = {}

    # ---- Step 2: analyze each table individually ----
    for tname, df in tables.items():
        detail = {
            "is_valid": True,
            "reasons": [],
            "duplicate_info": {},
            "duplicate_examples": {},
            "missing_mandatory": [],
        }

        if df is None or not isinstance(df, pd.DataFrame):
            detail["is_valid"] = False
            detail["reasons"].append("Unreadable or missing")
            validation["invalid_count"] += 1
            validation["details"][tname] = detail
            validation["files_skipped"].append({"file": tname, "reason": "Unreadable or missing"})
            continue

        if df.shape[0] == 0:
            detail["is_valid"] = False
            detail["reasons"].append("No rows found")
            validation["invalid_count"] += 1
            validation["details"][tname] = detail
            validation["files_skipped"].append({"file": tname, "reason": "No rows found"})
            continue

        pk_info = infer_pk(df)
        pks[tname] = pk_info

        dup_info: Dict[str, Any] = {}
        dup_examples: Dict[str, Any] = {}

        if pk_info.get("single"):
            pkcol = pk_info["single"]
            pk_res = check_pk_uniqueness(df, pkcol)
            if pk_res.get("dup_count_total", 0) > 0:
                dup_info[pkcol] = pk_res["dup_count_total"]
                dup_examples[pkcol] = pk_res["dup_values"]
                detail["reasons"].append(
                    f"Duplicate PK values found in {pkcol}: {pk_res['dup_count_total']}"
                )
        elif pk_info.get("composite"):
            cols = pk_info["composite"]
            dup_mask_any = df.duplicated(subset=cols, keep=False)
            dup_count_total = int(df.duplicated(subset=cols).sum())
            if dup_count_total > 0:
                keyname = "|".join(cols)
                grouped = df[dup_mask_any].groupby(cols)
                vals: Dict[str, Any] = {}
                for k, group in grouped:
                    if len(group) > 1:
                        valname = "||".join([str(x) for x in k]) if isinstance(k, tuple) else str(k)
                        vals[valname] = {
                            "count": len(group),
                            "examples": group.head(5).to_dict(orient="records"),
                        }
                dup_info[keyname] = dup_count_total
                dup_examples[keyname] = vals
                detail["reasons"].append(
                    f"Duplicate composite PK values found in {cols}: {dup_count_total}"
                )
        else:
            detail["reasons"].append(
                "No clear primary key found; consider adding a stable identifier"
            )

        stats = basic_column_stats(df)
        missing: List[str] = []
        for col, st in stats.items():
            if st["pct_null"] == 1.0:
                missing.append(col)
            elif st["pct_null"] >= 0.5:
                detail["reasons"].append(
                    f"Column '{col}' has >=50% null values ({st['pct_null']*100:.0f}%)"
                )

        detail["duplicate_info"] = dup_info
        detail["duplicate_examples"] = dup_examples
        detail["missing_mandatory"] = missing

        validation["validated_count"] += 1
        validation["details"][tname] = detail
        validation["files_analyzed"].append(tname)

    # ---- Step 3: infer FK relationships ----
    relationships = find_fk_candidates(tables, pks, subset_threshold=0.25)

    relation_type_counts: Dict[str, int] = {}
    relation_type_by_table: Dict[str, List[Dict[str, Any]]] = {}

    for r in relationships:
        typ = r.get("cardinality", "Unknown")
        relation_type_counts[typ] = relation_type_counts.get(typ, 0) + 1
        relation_type_by_table.setdefault(r["from"], []).append(r)
        relation_type_by_table.setdefault(r["to"], []).append(r)

    # Orphan detection (uses new logic)
    orphans = detect_orphans(tables, relationships)

    # ---- Step 4: logical / near duplicates ----
    logical_duplicates: Dict[str, Any] = {}
    near_duplicates: List[Dict[str, Any]] = []
    parent_merge_map: Dict[str, str] = {}

    for tname, df in tables.items():
        pkcol = pks.get(tname, {}).get("single")
        if not pkcol:
            continue

        ld = detect_logical_duplicates(df, pkcol)
        if ld["groups"]:
            logical_duplicates[tname] = ld["groups"]
            for dup_pk, can in ld["mapping"].items():
                parent_merge_map[dup_pk] = can

        nd = detect_near_duplicates(df, pkcol, numeric_tolerance=0.05)
        if nd:
            near_duplicates.extend([{"table": tname, **n} for n in nd])

    child_fk_conflicts: List[Dict[str, Any]] = []
    for rel in relationships:
        child = rel["from"]
        fk_col = rel["fk"]
        if child not in tables:
            continue
        child_df = tables[child]
        if fk_col not in child_df.columns:
            continue

        for idx, val in child_df[fk_col].astype(str).fillna("").items():
            if not val:
                continue
            if val in parent_merge_map:
                canonical = parent_merge_map[val]
                child_fk_conflicts.append(
                    {
                        "child_table": child,
                        "child_row_index": int(idx),
                        "fk_column": fk_col,
                        "fk_value": val,
                        "suggested_map_to": canonical,
                        "reason": f"Parent value '{val}' is a logical duplicate of canonical '{canonical}'",
                    }
                )

    # ---- Step 5: PK / full-row / business-key / FK consistency / 1:1 integrity ----
    pk_uniqueness_issues: Dict[str, Any] = {}
    full_row_dupes: Dict[str, Any] = {}
    business_key_issues: Dict[str, Any] = {}

    if business_key_map is None:
        business_key_map = {}

    for tname, df in tables.items():
        pkcol = pks.get(tname, {}).get("single")

        if pkcol:
            pk_res = check_pk_uniqueness(df, pkcol)
            if pk_res.get("dup_count_total", 0) > 0:
                pk_uniqueness_issues[tname] = pk_res

        fr = check_full_row_uniqueness(df)
        if fr["total_groups"] > 0:
            full_row_dupes[tname] = fr

        bk_defs = business_key_map.get(tname, [])
        if not bk_defs:
            stats = basic_column_stats(df)
            ranked_cols = sorted(stats.items(), key=lambda kv: kv[1]["pct_unique"], reverse=True)
            candidate_cols = [c for c, s in ranked_cols[:6]]
            bk_sets = [list(combo) for combo in itertools.combinations(candidate_cols, 2)]
            bk_defs = bk_sets[:6]

        bk_check = check_business_key_uniqueness(df, bk_defs)
        bk_viol = [b for b in bk_check["checked_keys"] if b.get("dup_count", 0) > 0]
        if bk_viol:
            business_key_issues[tname] = bk_viol

    expanded_orphans = check_referential_integrity(tables, relationships, pks)
    fk_merge_conflicts = check_fk_consistency_for_merged_parents(tables, relationships, parent_merge_map)
    one_to_one_issues = check_one_to_one_integrity(tables, relationships)

    # ---- Step 6: RCA suggestions ----
    rca_suggestions: List[Dict[str, Any]] = []
    for tname, pk in pks.items():
        if not pk.get("single") and not pk.get("composite"):
            rca_suggestions.append(
                {
                    "area": f"Primary Key: {tname}",
                    "priority": "High",
                    "reason": "No clear primary key found. RCA model requires stable identifiers.",
                    "actions": [
                        "Add a synthetic ID or identify a natural unique key",
                        "Ensure column is populated and unique",
                    ],
                }
            )

    for r in relationships[:50]:
        if r.get("overlap_prop", 0) >= 0.9 and r.get("matched_values_count", 0) >= 3:
            rca_suggestions.append(
                {
                    "area": f"Foreign Key: {r['from']}.{r['fk']} -> {r['to']}",
                    "priority": "Medium",
                    "reason": "High overlap between child FK values and parent PK candidate",
                    "actions": [
                        "Validate semantics",
                        "Consider referential integrity enforcement",
                    ],
                }
            )

    # ---- Step 7: Build main report ----
    report: Dict[str, Any] = {
        "validation": validation,
        "relationships": relationships,
        "relation_type_counts": relation_type_counts,
        "relation_type_by_table": relation_type_by_table,
        "orphans": orphans,
        "logical_duplicates": logical_duplicates,
        "near_duplicates": near_duplicates,
        "parent_merge_map": parent_merge_map,
        "child_fk_conflicts": child_fk_conflicts,
        "pk_uniqueness_issues": pk_uniqueness_issues,
        "full_row_duplicates": full_row_dupes,
        "business_key_issues": business_key_issues,
        "expanded_orphans": expanded_orphans,
        "fk_merge_conflicts": fk_merge_conflicts,
        "one_to_one_issues": one_to_one_issues,
        "rca_suggestions": rca_suggestions,
    }

    # ---- Step 8: remediation artifacts ----
    fk_updates_path = os.path.join(output_folder, "fk_updates.sql")
    fk_updates: List[str] = []
    child_rows_to_fix_rows: List[Dict[str, Any]] = []

    for c in fk_merge_conflicts.get("conflicts", []):
        child = c["child_table"]
        fk_col = c["fk_column"]
        canon = c["canonical"]
        dup_vals = [d["dup_val"] for d in c["duplicate_values"]]
        sql = c["suggested_updates"][0]
        fk_updates.append(sql)

        if child in tables:
            child_df = tables[child]
            mask = child_df[fk_col].astype(str).isin([str(v) for v in dup_vals])
            subset = child_df.loc[mask]
            if not subset.empty:
                subset_copy = subset.copy()
                subset_copy["_suggested_new_fk"] = canon
                child_rows_to_fix_rows.extend(subset_copy.to_dict(orient="records"))

    if fk_updates:
        with open(fk_updates_path, "w", encoding="utf-8") as f:
            f.write("\n".join(fk_updates))

    fk_fix_csv = os.path.join(output_folder, "child_rows_to_fix.csv")
    if child_rows_to_fix_rows:
        pd.DataFrame(child_rows_to_fix_rows).to_csv(fk_fix_csv, index=False)

    remediation = {
        "fk_updates_sql": fk_updates_path if fk_updates else None,
        "child_rows_to_fix_csv": fk_fix_csv if child_rows_to_fix_rows else None,
        "fk_merge_conflicts_summary": fk_merge_conflicts.get("summary", {}),
    }
    report["remediation_artifacts"] = remediation

    # ---- Step 9: save report JSON ----
    report_path = os.path.join(output_folder, "analysis_report.json")
    save_report_json(report, report_path)

    return report
