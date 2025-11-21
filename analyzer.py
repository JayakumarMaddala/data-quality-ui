# analyzer.py
"""
Final analyzer for CPQ -> RCA migration.
- file reading (CSV/Excel)
- validation checks (duplicates, mandatory fields)
- FK detection heuristics
- relationship cardinality inference
- orphan detection (fixed Unalignable boolean Series issue)
- report generation (JSON-serializable)
"""
from typing import Dict, List, Any
import pandas as pd
import re
import json
from collections import defaultdict

# --- reading ---------------------------------------------------------------
def read_table(path_or_file) -> pd.DataFrame:
    """
    Reads CSV or Excel path or file-like object. Normalizes empty strings to NaN.
    """
    if hasattr(path_or_file, "read"):
        # file-like object (uploaded by Streamlit)
        name = getattr(path_or_file, "name", "")
        ext = name.lower().split(".")[-1] if "." in name else ""
    else:
        p = str(path_or_file)
        ext = p.lower().split(".")[-1] if "." in p else ""

    if ext in ("csv",):
        df = pd.read_csv(path_or_file, dtype=str, keep_default_na=False, na_values=[""])
    else:
        # treat as excel if not csv
        df = pd.read_excel(path_or_file, dtype=str, keep_default_na=False, na_values=[""])
    df = df.replace({"": pd.NA})
    # strip string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()
    return df

# --- heuristics ------------------------------------------------------------
_id_patterns = re.compile(r'(^|_)id$|_id$|Id$|ID$', re.IGNORECASE)
_fk_like_patterns = re.compile(r'(^|_)id$|_id$|Id$|ID$|^.*__r$|^.*__c$', re.IGNORECASE)

def candidate_key_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if _id_patterns.search(c):
            cols.append(c)
    for c in ["Id", "ID", "Id__c", "Name", "ProductCode", "PRODUCT2_ID"]:
        if c in df.columns and c not in cols:
            cols.append(c)
    # add any fully-unique column
    for c in df.columns:
        try:
            nonnull = df[c].notna().sum()
            if nonnull > 0 and df[c].nunique(dropna=True) == nonnull:
                if c not in cols:
                    cols.append(c)
        except Exception:
            pass
    return cols

def find_fk_candidates(df: pd.DataFrame) -> List[str]:
    cands = []
    for c in df.columns:
        try:
            if _fk_like_patterns.search(c) and df[c].notna().any():
                cands.append(c)
        except Exception:
            pass
    return cands

# --- validation ------------------------------------------------------------
def validate_table(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns:
    {
      is_valid: bool,
      reasons: [...],
      duplicate_info: {pk_col: dup_count},
      missing_mandatory: [...]
    }
    """
    if df is None:
        return {"is_valid": False, "reasons": ["failed_to_load"], "duplicate_info": {}, "missing_mandatory": []}
    if not isinstance(df, pd.DataFrame):
        return {"is_valid": False, "reasons": ["not_a_dataframe"], "duplicate_info": {}, "missing_mandatory": []}

    reasons = []
    dup_info = {}
    missing_mand = []

    if df.shape[0] == 0:
        reasons.append("no_rows")
    if df.shape[1] == 0:
        reasons.append("no_columns")

    # duplicates check on candidate PKs
    pks = candidate_key_columns(df)
    for pk in pks:
        if pk in df.columns:
            total = int(df[pk].notna().sum())
            unique = int(df[pk].nunique(dropna=True))
            dup = total - unique
            if dup > 0:
                dup_info[pk] = dup
                reasons.append(f"duplicates_in_{pk}")

    # mandatory fields heuristic
    mandatory_candidates = ["Id", "Name", "ProductCode", "SBQQ__Quote__c", "SBQQ__Product__c"]
    for m in mandatory_candidates:
        if m in df.columns:
            nulls = int(df[m].isna().sum())
            # if more than 50% null, consider problem
            if df.shape[0] > 0 and (nulls / df.shape[0]) > 0.5:
                missing_mand.append(m)
                reasons.append(f"many_nulls_in_{m}")
        else:
            # require Id and Name strongly
            if m in ("Id", "Name"):
                missing_mand.append(m)
                reasons.append(f"missing_{m}")

    is_valid = len(reasons) == 0
    return {"is_valid": is_valid, "reasons": reasons, "duplicate_info": dup_info, "missing_mandatory": missing_mand}

# --- relationships --------------------------------------------------------
def infer_relationships(tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Heuristic FK detection and cardinality inference.
    Returns list of relationship dicts.
    """
    rels = []
    if not tables:
        return rels

    # precompute candidate PKs
    keys = {name: candidate_key_columns(df) for name, df in tables.items() if isinstance(df, pd.DataFrame)}

    for child_name, child_df in tables.items():
        if child_df is None or not isinstance(child_df, pd.DataFrame):
            continue
        fk_cols = find_fk_candidates(child_df)
        for fk in fk_cols:
            try:
                fk_vals = child_df[fk].dropna().astype(str)
            except Exception:
                fk_vals = pd.Series([], dtype=str)
            if fk_vals.empty:
                continue

            matched_table = None
            matched_pk = None

            # try name hint first
            for parent_name, parent_df in tables.items():
                if parent_name == child_name or parent_df is None:
                    continue
                if parent_name.lower() in fk.lower() or parent_name.split("__")[0].lower() in fk.lower():
                    for pk in keys.get(parent_name, []):
                        if pk in parent_df.columns:
                            parent_vals = parent_df[pk].dropna().astype(str)
                            overlap = len(set(fk_vals.unique()) & set(parent_vals.unique()))
                            if overlap > 0:
                                matched_table = parent_name
                                matched_pk = pk
                                break
                if matched_table:
                    break

            # fallback by value overlap
            if not matched_table:
                best = (None, None, 0)
                for parent_name, parent_df in tables.items():
                    if parent_name == child_name or parent_df is None:
                        continue
                    for pk in keys.get(parent_name, []):
                        if pk not in parent_df.columns:
                            continue
                        parent_vals = parent_df[pk].dropna().astype(str)
                        overlap = len(set(fk_vals.unique()) & set(parent_vals.unique()))
                        if overlap > best[2]:
                            best = (parent_name, pk, overlap)
                if best[2] > 0:
                    matched_table, matched_pk, _ = best

            if not matched_table:
                rels.append({
                    "from": child_name,
                    "fk": fk,
                    "to": None,
                    "to_pk_candidate": None,
                    "type": "Unknown",
                    "stats": {"distinct_fk_values": int(fk_vals.nunique(dropna=True)), "rows_with_fk": int(fk_vals.shape[0])}
                })
                continue

            # cardinality heuristics (avoid using mismatched index masks)
            parent_df = tables[matched_table]
            fk_series = child_df[fk].astype(str) if fk in child_df.columns else pd.Series([], dtype=str)
            fk_nonnull = fk_series.dropna().astype(str)
            parent_referenced_counts = fk_nonnull.value_counts()
            parent_multi_count = int((parent_referenced_counts > 1).sum())
            parent_single_count = int((parent_referenced_counts == 1).sum())

            if parent_multi_count > 0:
                rel_type = "ManyToOne"
            elif parent_single_count == fk_nonnull.shape[0] and fk_nonnull.shape[0] > 0:
                rel_type = "OneToOne"
            else:
                rel_type = "Unknown"

            rels.append({
                "from": child_name,
                "fk": fk,
                "to": matched_table,
                "to_pk_candidate": matched_pk,
                "type": rel_type,
                "stats": {
                    "parent_rows": int(parent_df.shape[0]),
                    "child_rows": int(child_df.shape[0]),
                    "rows_with_fk_nonnull": int(fk_nonnull.shape[0]),
                    "distinct_fk_values": int(fk_nonnull.nunique(dropna=True)),
                    "examples_fk_values": list(fk_nonnull.unique()[:5])
                }
            })

    # detect junctions (many-to-many) - mark related rels
    for table_name, df in tables.items():
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        fk_cols = find_fk_candidates(df)
        resolved_targets = []
        for fk in fk_cols:
            for r in rels:
                if r["from"] == table_name and r["fk"] == fk and r.get("to"):
                    resolved_targets.append((fk, r["to"]))
        targets = set([t for (_, t) in resolved_targets])
        if len(targets) >= 2:
            pk_cands = candidate_key_columns(df)
            unique_pk_found = False
            for pk in pk_cands:
                if pk in df.columns:
                    if df[pk].notna().sum() > 0 and df[pk].nunique(dropna=True) == df[pk].notna().sum():
                        unique_pk_found = True
            if not unique_pk_found:
                for fk, parent in resolved_targets:
                    for r in rels:
                        if r["from"] == table_name and r["fk"] == fk and r.get("to") == parent:
                            r["type"] = "Junction (ManyToMany)"
                rels.append({
                    "from": table_name,
                    "fk": None,
                    "to": list(targets),
                    "to_pk_candidate": None,
                    "type": "ManyToMany (junction)",
                    "stats": {"fk_columns": [f for f, _ in resolved_targets], "row_count": int(df.shape[0])}
                })

    return rels

# --- orphan detection (FIX for unalignable boolean series) -----------------
def detect_orphans(child_df: pd.DataFrame, parent_df: pd.DataFrame, child_fk_col: str, parent_pk_col: str, examples=5):
    """
    Returns orphan count and examples. Fixed to avoid boolean mask alignment issues:
    - build boolean mask aligned to child_df index explicitly.
    """
    if child_df is None or parent_df is None:
        return {"count": 0, "examples": []}
    if child_fk_col not in child_df.columns or parent_pk_col not in parent_df.columns:
        return {"count": 0, "examples": []}

    # create sets of parent PK values as strings
    parent_vals = set(parent_df[parent_pk_col].dropna().astype(str).unique())
    # create a mask aligned with child_df index
    fk_series_full = child_df[child_fk_col].astype("string")
    # rows where fk is not null and not present in parent_vals
    mask = fk_series_full.notna() & (~fk_series_full.isin(parent_vals))
    missing = child_df.loc[mask]
    return {"count": int(missing.shape[0]), "examples": missing.head(examples).to_dict(orient="records")}

# --- report generation -----------------------------------------------------
def generate_analysis_report(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    report = {
        "summary": {},
        "tables": {},
        "validation": {},
        "relationships": [],
        "relation_type_counts": {},
        "orphans": [],
        "rca_suggestions": []
    }

    # tables metadata
    for name, df in tables.items():
        report["tables"][name] = {
            "row_count": int(df.shape[0]) if hasattr(df, "shape") else 0,
            "col_count": int(df.shape[1]) if hasattr(df, "shape") else 0,
            "columns": list(df.columns) if hasattr(df, "columns") else [],
            "sample_rows": df.head(3).to_dict(orient="records") if hasattr(df, "head") else []
        }

    # validation
    uploaded_count = len(tables)
    validated_count = 0
    validation_details = {}
    for name, df in tables.items():
        v = validate_table(df)
        validation_details[name] = v
        if v.get("is_valid"):
            validated_count += 1

    report["validation"] = {
        "uploaded_count": uploaded_count,
        "validated_count": validated_count,
        "invalid_count": uploaded_count - validated_count,
        "details": validation_details
    }

    # relationships (use all dataframes that are proper)
    processed_tables = {n: df for n, df in tables.items() if isinstance(df, pd.DataFrame)}
    rels = infer_relationships(processed_tables)
    report["relationships"] = rels

    # orphans detection
    orphans = []
    for r in rels:
        if r.get("to") and r.get("to_pk_candidate") and r.get("fk"):
            child_df = processed_tables.get(r["from"])
            parent_df = processed_tables.get(r["to"])
            if child_df is not None and parent_df is not None:
                o = detect_orphans(child_df, parent_df, r["fk"], r["to_pk_candidate"])
                if o["count"] > 0:
                    orphans.append({
                        "from": r["from"],
                        "to": r["to"],
                        "fk": r["fk"],
                        "parent_pk": r["to_pk_candidate"],
                        "count": o["count"],
                        "examples": o["examples"]
                    })
    report["orphans"] = orphans

    # relation type counts
    type_counts = defaultdict(int)
    for r in rels:
        t = r.get("type", "Unknown")
        type_counts[t] += 1
    report["relation_type_counts"] = dict(type_counts)

    # simple RCA suggestions
    suggestions = []
    if any("Product" in k for k in processed_tables.keys()):
        suggestions.append({
            "area": "Product Catalog / Product2",
            "priority": "High",
            "reason": "Migrate product metadata before pricing/quotes.",
            "actions": ["Migrate Product2 and PricebookEntry first.", "Ensure pricebook entries for currencies exist."]
        })
    if any("ManyToMany" in t for t in type_counts.keys()):
        suggestions.append({
            "area": "Many-to-Many / Junction",
            "priority": "Medium",
            "reason": "Detected junction-like objects; map to association tables.",
            "actions": ["Treat as junction association entity or denormalize if warrants."]
        })

    report["rca_suggestions"] = suggestions

    report["summary"]["table_count"] = uploaded_count
    report["summary"]["relationship_count"] = len(rels)
    report["summary"]["orphan_issues"] = len(orphans)

    return report

# convenience save
def save_report_json(report: Dict[str, Any], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
