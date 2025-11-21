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

# image path (user-provided screenshot in the container)
SCREENSHOT_PATH = "/mnt/data/c4bdf6a5-60db-460f-ace0-83ce071b9bf1.png"

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

# ---------------- Helper for badges ----------------
def badge_html(text: str, color_hex: str = "#d4f7dc", text_color: str = "#085f2e"):
    # small pill-like badge
    html = f"""
    <div style="
        display:inline-block;
        padding:8px 12px;
        border-radius:8px;
        background:{color_hex};
        color:{text_color};
        font-weight:600;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin:4px 0;
    ">{text}</div>
    """
    return html

def status_to_badge(status: str, reason: str = ""):
    s = status.lower()
    if s in ("analyzed","valid","ok"):
        return badge_html("Analyzed", "#e6f9ee", "#0b6b36")
    if s in ("skipped","invalid"):
        # include reason
        label = "Skipped" if s == "skipped" else status.capitalize()
        badge = badge_html(label, "#fff6d9", "#7a5a00")
        if reason:
            reason_html = f"<div style='color:#6b6b6b;margin-top:6px;font-size:13px'>Reason: {st.markdown(reason, unsafe_allow_html=False) or reason}</div>"
            # We can't easily return mixed st.markdown content here, so just attach reason text (we'll display separately)
            return badge
        return badge
    # default
    return badge_html(status, "#eef2ff", "#24346b")

# ---------------- Analyzer UI ----------------
def analyzer_ui(username: str):
    st.title(f"CPQ → RCA Analyzer (User: {username})")
    logout_button()

    # show provided screenshot if exists (example visual)
    if os.path.exists(SCREENSHOT_PATH):
        st.markdown("**Sample UI / Summary Preview**")
        st.image(SCREENSHOT_PATH, use_column_width=True)
    else:
        st.info("No sample screenshot found in container path.")

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
    # Top-level summary (counts)
    st.markdown("## Summary")
    validation = report.get("validation", {})
    uploaded = validation.get("uploaded_count", 0)
    valid = validation.get("validated_count", 0)
    skipped = validation.get("skipped_count", 0) or validation.get("invalid_count", 0)
    inferred_rels = len(report.get("relationships", []))
    orphan_issues = len(report.get("orphans", []))

    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    col1.metric("Tables uploaded", uploaded)
    col2.metric("Tables analyzed (valid for analysis)", valid)
    col3.metric("Tables skipped", skipped)
    col4.metric("Inferred relationships", inferred_rels)
    col5.metric("Orphan relationship issues", orphan_issues)

    st.markdown("---")
    st.markdown("## Uploaded file status")
    st.markdown("**Filename — Status — Reason (if skipped)**")

    details = validation.get("details", {})
    if not details:
        st.write("No file-level validation details available.")
    else:
        # render rows similar to screenshot
        for fname, vd in details.items():
            st.write("")  # spacer
            left, right = st.columns([3,1])
            left.markdown(f"**{fname}**")
            # determine status label & reason
            is_valid = vd.get("is_valid", False)
            reasons = vd.get("reasons", [])
            reason_text = "; ".join(reasons) if reasons else ""
            duplicate_info = vd.get("duplicate_info", {})
            if is_valid:
                right.markdown(badge_html("Analyzed", "#e6f9ee", "#0b6b36"), unsafe_allow_html=True)
            else:
                # treat as skipped/invalid
                right.markdown(badge_html("Skipped", "#fff6d9", "#7a5a00"), unsafe_allow_html=True)
                if reason_text:
                    # small muted reason at right of row
                    right.markdown(f"<div style='font-size:12px;color:#5b5b5b;margin-top:6px'>Reason: {reason_text}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Skipped files (with reasons)")
    skipped_rows = []
    for fname, vd in details.items():
        if not vd.get("is_valid", False):
            reasons = vd.get("reasons", [])
            skipped_rows.append({"file": fname, "reasons": "; ".join(reasons) or "Unknown"})
    if skipped_rows:
        st.table(pd.DataFrame(skipped_rows))
    else:
        st.write("No skipped files.")

    st.markdown("---")
    # Relationships sample and other sections (keep similar to previous UI but improved)
    st.subheader("Inferred Relationships (sample)")
    rels = report.get("relationships", [])
    if rels:
        rel_df = pd.DataFrame(rels).fillna("-")
        show_cols = [c for c in ["from","fk","to","to_pk_candidate","type"] if c in rel_df.columns]
        st.dataframe(rel_df[show_cols], height=240)
    else:
        st.write("No relationships inferred.")

    st.subheader("Orphan records (examples)")
    orphans = report.get("orphans", [])
    if orphans:
        for o in orphans:
            st.write(f"Child: **{o.get('from','?')}** → Parent: **{o.get('to','?')}** FK: `{o.get('fk','?')}` missing {o.get('count',0)} rows")
            st.json(o.get("examples", []))
    else:
        st.success("No orphan records detected for inferred relationships.")

    st.subheader("RCA Mapping Suggestions")
    for s in report.get("rca_suggestions", []):
        st.markdown(f"**{s.get('area','Unknown')}** — _{s.get('priority','')}_")
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
