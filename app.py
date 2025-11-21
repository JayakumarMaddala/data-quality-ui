# app.py
import streamlit as st
from analyzer import read_table, generate_analysis_report, save_report_json
import pandas as pd
import json
import datetime
import os
from typing import Dict, Any, List, Tuple
import auth_db

# persistent dir (local default sample_data; on Render set env PERSISTENT_DIR=/persistent)
PERSISTENT_DIR = os.getenv("PERSISTENT_DIR", os.path.join(os.path.dirname(__file__), "sample_data"))
os.makedirs(PERSISTENT_DIR, exist_ok=True)

st.set_page_config(page_title="CPQ -> RCA Analyzer (Jay)", layout="wide")
auth_db.init_db()

if "user" not in st.session_state:
    st.session_state["user"] = None

# ---------------- AUTH UI ----------------
def login_screen():
    st.title("🔐 CPQ → RCA Analyzer - Login Required [Jay]")
    tab1, tab2 = st.tabs(["Login", "Create Account"])

    with tab1:
        st.subheader("Login")
        user_in = st.text_input("Username or Email", key="login_user_input")
        pwd_in = st.text_input("Password", type="password", key="login_pwd_input")
        if st.button("Login", key="login_btn"):
            ok, username = auth_db.verify_user(user_in.strip(), pwd_in)
            if ok:
                st.session_state["user"] = username
                st.success(f"Logged in as {username}")
                st.rerun()
            else:
                st.error("Invalid username/email or password")

    with tab2:
        st.subheader("Create New Account")
        new_user = st.text_input("Choose Username", key="signup_user_input")
        new_email = st.text_input("Email", key="signup_email_input")
        new_pwd = st.text_input("Password", type="password", key="signup_pwd_input")
        new_pwd2 = st.text_input("Confirm Password", type="password", key="signup_pwd2_input")
        if st.button("Create Account", key="signup_btn"):
            if new_pwd != new_pwd2:
                st.error("Passwords do not match")
            elif not new_user or not new_email or not new_pwd:
                st.error("All fields are required")
            else:
                ok, msg = auth_db.create_user(new_user.strip(), new_email.strip(), new_pwd)
                if ok:
                    st.success("Account created! You can now login.")
                else:
                    st.error(msg)

def logout_button():
    if st.button("Logout"):
        st.session_state["user"] = None
        st.success("Logged out")
        st.rerun()

# ---------------- Analyzer UI ----------------
def analyzer_ui(username: str):
    st.title(f"CPQ → RCA Analyzer (User: {username})")
    logout_button()

    # show generic sample pdf if exists
    sample_pdf_path = os.path.join(PERSISTENT_DIR, "cpq_reference.pdf")
    if os.path.exists(sample_pdf_path):
        st.markdown("📄 **Sample Reference PDF available**")
        st.code(sample_pdf_path)
    else:
        st.info("No sample reference PDF found in persistent folder (place cpq_reference.pdf there).")

    st.write("Upload CPQ CSV/Excel exports (Product, Pricebook, Quote, QuoteLine, PricebookEntry, etc.).")

    uploaded_files = st.file_uploader("Upload CPQ object files", type=["csv","xlsx","xls"], accept_multiple_files=True)

    # quick load sample_data button (useful for local testing)
    if st.button("Load all files from sample_data"):
        import glob
        sample_paths = glob.glob(os.path.join(PERSISTENT_DIR, "*.xlsx")) + glob.glob(os.path.join(PERSISTENT_DIR, "*.csv"))
        tables = {}
        for p in sample_paths:
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                tables[name] = read_table(p)
            except Exception as e:
                tables[name] = None
        report = generate_analysis_report(tables)
        _save_and_render(report, username)
        return

    if not uploaded_files:
        st.info("Upload files to begin analysis (or use 'Load all files from sample_data').")
        return

    # read uploaded files
    tables: Dict[str, Any] = {}
    read_errors: List[Tuple[str, str]] = []
    for f in uploaded_files:
        name = os.path.splitext(f.name)[0]
        try:
            df = read_table(f)
            tables[name] = df
        except Exception as e:
            tables[name] = None
            read_errors.append((name, str(e)))
            st.error(f"Failed to read {f.name}: {e}")

    st.info(f"{len(uploaded_files)} file(s) uploaded.")
    with st.spinner("Running analysis..."):
        report = generate_analysis_report(tables)

    _save_and_render(report, username)

def _save_and_render(report: Dict[str, Any], username: str):
    # save to persistent dir per-user
    user_reports_dir = os.path.join(PERSISTENT_DIR, "reports", username)
    os.makedirs(user_reports_dir, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    file_path = os.path.join(user_reports_dir, f"report_{timestamp}.json")
    save_report_json(report, file_path)
    st.success(f"Report saved: {file_path}")
    render_report_ui(report)

def render_report_ui(report: Dict[str, Any]):
    st.subheader("Validation Overview")
    validation = report.get("validation", {})
    uploaded = validation.get("uploaded_count", 0)
    valid = validation.get("validated_count", 0)
    invalid = validation.get("invalid_count", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Files uploaded", uploaded)
    c2.metric("Files validated", valid)
    c3.metric("Files invalid", invalid)

    st.subheader("Validation details (per file)")
    details = validation.get("details", {})
    rows = []
    for fname, vd in details.items():
        rows.append({
            "file": fname,
            "is_valid": vd.get("is_valid", False),
            "reasons": "; ".join(vd.get("reasons", [])),
            "duplicates": json.dumps(vd.get("duplicate_info", {})),
            "missing_mandatory": ", ".join(vd.get("missing_mandatory", []))
        })
    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.write("No validation details available.")

    st.subheader("Relation type breakdown")
    rtc = report.get("relation_type_counts", {})
    if rtc:
        rtc_df = pd.DataFrame([{"relation_type": k, "count": v} for k, v in rtc.items()]).sort_values("count", ascending=False)
        st.table(rtc_df)
    else:
        st.write("No relationships inferred.")

    st.subheader("Inferred Relationships (sample)")
    rels = report.get("relationships", [])
    if rels:
        rel_df = pd.DataFrame(rels).fillna("-")
        show_cols = [c for c in ["from","fk","to","to_pk_candidate","type"] if c in rel_df.columns]
        st.dataframe(rel_df[show_cols], height=250)
    else:
        st.write("No relationships inferred.")

    st.subheader("Duplicates summary")
    dup_rows = []
    for fname, vd in details.items():
        dup = vd.get("duplicate_info", {})
        if dup:
            for pk_col, cnt in dup.items():
                dup_rows.append({"file": fname, "pk_col": pk_col, "duplicate_count": cnt})
    if dup_rows:
        st.table(pd.DataFrame(dup_rows))
    else:
        st.write("No duplicate primary-key issues detected.")

    st.subheader("Orphan records (examples)")
    orphans = report.get("orphans", [])
    if orphans:
        for o in orphans:
            st.write(f"Child: **{o['from']}** → Parent: **{o['to']}** FK: `{o['fk']}` missing {o['count']} rows")
            st.json(o.get("examples", []))
    else:
        st.success("No orphan records detected for inferred relationships.")

    st.subheader("RCA Mapping Suggestions")
    for s in report.get("rca_suggestions", []):
        st.markdown(f"**{s['area']}** — _{s.get('priority','')}_")
        st.write(s.get("reason",""))
        for a in s.get("actions", []):
            st.write(f"- {a}")

    st.subheader("Download full JSON report")
    st.download_button("Download report as JSON", json.dumps(report, indent=2), file_name=f"analysis_report_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json", mime="application/json")

# main
if st.session_state["user"] is None:
    login_screen()
else:
    analyzer_ui(st.session_state["user"])
