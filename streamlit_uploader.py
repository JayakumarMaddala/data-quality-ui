# streamlit_uploader.py
import streamlit as st
import os
import io
import pandas as pd
from analyzer import read_table, generate_analysis_report, load_business_key_map
import json

def normalize_name(name: str) -> str:
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]
    base = base.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in base:
        base = base.replace("__", "_")
    return base

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    df = df.copy()
    df.columns = new_cols
    return df

st.set_page_config(page_title="CPQ → RCA Analyzer (Uploader)", layout="wide")
st.title("CPQ → RCA Analyzer (Uploader)")

uploaded = st.file_uploader("Upload CSV / Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)
bkmap_file = st.file_uploader("Optional: business key map (JSON/YAML)", type=["json", "yml", "yaml"], accept_multiple_files=False)
out_folder = st.text_input("Output folder", value="analysis_output")

if st.button("Run analysis on uploaded files"):
    if not uploaded:
        st.warning("Please upload files.")
    else:
        os.makedirs(out_folder, exist_ok=True)
        tables = {}
        for f in uploaded:
            key = normalize_name(f.name)
            try:
                content = f.read()
                obj = read_table(content)
                if isinstance(obj, dict):
                    for sheet, df in obj.items():
                        tables[f"{key}__{normalize_name(sheet)}"] = normalize_dataframe_columns(df)
                else:
                    tables[key] = normalize_dataframe_columns(obj)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")

        bkmap = {}
        if bkmap_file:
            try:
                bkmap_path = os.path.join(out_folder, bkmap_file.name)
                with open(bkmap_path, "wb") as wf:
                    wf.write(bkmap_file.read())
                bkmap = load_business_key_map(bkmap_path)
            except Exception as e:
                st.warning(f"Failed to load bkmap: {e}")

        with st.spinner("Analyzing..."):
            report = generate_analysis_report(tables, business_key_map=bkmap, output_folder=out_folder)

        st.success("Analysis complete")
        st.json({
            "pk_issues": report.get("pk_uniqueness_issues", {}),
            "full_row_dupes": report.get("full_row_duplicates", {}),
            "logical_dupes": report.get("logical_duplicates", {}),
            "fk_conflicts": report.get("fk_merge_conflicts", {})
        })

        # download full report
        report_path = os.path.join(out_folder, "analysis_report.json")
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button("Download analysis_report.json", f, file_name="analysis_report.json")
