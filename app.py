# app_v2.py
"""
Streamlit app with:
 - Login card (local auth_db)
 - CSV / XLSX upload and analysis via analyzer.generate_analysis_report
 - Salesforce quick username+token connect (Test Connection button with full traceback)
 - Salesforce OAuth Connected App using PKCE: build auth URL -> paste code -> exchange for tokens
 - Ability to list SBQQ__* objects and fetch sample rows into analyzer

This version (v2) adds:
 - Two-step flow:
     1) Upload page
     2) Analysis page
 - On upload page:
     * Show files, Salesforce connect, business key map
     * Button: "Click here for the Analysis"
 - On analysis page:
     * Uses cached report (generated once)
     * 5 sections, each with its own "Run Analysis" button:
         1) Primary Key uniqueness
         2) Full-row duplicates
         3) Logical / Business key uniqueness
         4) FK Consistency (incl. 1:1 + FK merge conflicts)
         5) Referential Integrity (orphans)
"""

import yaml
import os
import json
import datetime
import traceback
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import streamlit as st
import requests

# local project modules (must exist)
import analyzer    # must provide read_table, generate_analysis_report, load_business_key_map
import auth_db     # must provide init_db(), verify_user(), create_user()

# PKCE helpers and token-exchange helpers (create sf_pkce_oauth.py as separate module)
from sf_pkce_oauth import (
    generate_code_verifier,
    code_challenge_from_verifier,
    build_salesforce_auth_url,
    exchange_code_for_token,
    sf_from_tokens as sf_from_tokens_helper,
)

# simple-salesforce quick username+token connect
from simple_salesforce import Salesforce, SalesforceLogin

# -------------------- Config --------------------
PERSISTENT_DIR = os.getenv(
    "PERSISTENT_DIR", os.path.join(os.path.dirname(__file__), "sample_data")
)
os.makedirs(PERSISTENT_DIR, exist_ok=True)

# Use the local path you uploaded earlier. This will be transformed to a url if needed.
SAMPLE_PDF_PATH = "/mnt/data/CPQ Book by Chandra.pdf"

st.set_page_config(page_title="CPQ → RCA Analyzer (Jay)", layout="wide")

# -------------------- Helpers --------------------


def sample_preview(rows, max_chars: int = 300) -> str:
    try:
        s = json.dumps(rows, indent=0)
    except Exception:
        s = str(rows)
    return (s[:max_chars] + "...") if len(s) > max_chars else s


def normalize_sf_name(n: str) -> str:
    return str(n).strip().lower().replace("/", "_").replace(" ", "_")


# -------------------- Salesforce helpers --------------------


def connect_salesforce_username_password(
    username: str,
    password: str,
    security_token: str,
    domain: str = "login",
) -> Salesforce:
    """
    Quick dev connect using SalesforceLogin.
    Raises exception on failure; Test Connection button will capture traceback.
    """
    session_id, instance = SalesforceLogin(
        username=username,
        password=password,
        security_token=security_token,
        domain=domain,
    )
    sf = Salesforce(instance=instance, session_id=session_id)
    return sf


def sf_from_tokens(instance_url: str, access_token: str) -> Salesforce:
    """Build simple_salesforce Salesforce object from OAuth tokens."""
    return sf_from_tokens_helper(instance_url=instance_url, access_token=access_token)


def fetch_org_sobjects(sf: Salesforce, limit: int = 500) -> List[str]:
    desc = sf.describe()
    return [o["name"] for o in desc.get("sobjects", [])][:limit]


def fetch_sample_records(
    sf: Salesforce,
    sobject_api_name: str,
    limit: int = 200,
) -> pd.DataFrame:
    try:
        d = sf.restful(f"sobjects/{sobject_api_name}/describe")
        fields = [f["name"] for f in d.get("fields", [])][:50]
        if not fields:
            return pd.DataFrame()
        soql = f"SELECT {', '.join(fields)} FROM {sobject_api_name} LIMIT {limit}"
        res = sf.query_all(soql)
        records = res.get("records", [])
        for r in records:
            r.pop("attributes", None)
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


# -------------------- UI: Salesforce connection panel (PKCE) --------------------


def salesforce_connect_panel(
    out_dir: str = "analysis_output",
) -> Tuple[Dict[str, pd.DataFrame], List[str], Optional[dict]]:
    """
    returns (sf_tables, saved_files, oauth_tokens)
     - sf_tables: dict name->DataFrame fetched via quick connect or OAuth
     - saved_files: list of local artifact paths saved
     - oauth_tokens: dict with token response if OAuth used
    """
    st.markdown("### Connect to Salesforce (optional)")
    st.info("Use OAuth (Connected App) (recommended).")

    col1, col2 = st.columns(2)

    sf_tables: Dict[str, pd.DataFrame] = {}
    saved_files: List[str] = []
    oauth_tokens = None

    # Ensure pkce-related storage exists
    if "pkce" not in st.session_state:
        st.session_state["pkce"] = {}
    if "pkce_auth_url" not in st.session_state:
        st.session_state["pkce_auth_url"] = {}
    if "prev_client_id" not in st.session_state:
        st.session_state["prev_client_id"] = None

    with col1:
        st.markdown("#### OAuth / Connected App (PKCE)")
        st.markdown(
            "Create a Connected App in Salesforce and supply Client ID/Secret + redirect URI."
        )
        st.caption(
            "Scopes: api and refresh_token. For Cloud use your deployed URL as redirect."
        )

        client_id = st.text_input(
            "Connected App Client ID (Consumer Key)", key="sf_client_id"
        )

        # ---- PKCE reset when Client ID changes ----
        prev_client_id = st.session_state.get("prev_client_id")
        if prev_client_id and prev_client_id != client_id:
            st.session_state["pkce"].pop(prev_client_id, None)
            st.session_state["pkce_auth_url"].pop(prev_client_id, None)
        st.session_state["prev_client_id"] = client_id
        # -------------------------------------------

        client_secret = st.text_input(
            "Connected App Client Secret (optional)",
            type="password",
            key="sf_client_secret",
        )

        redirect_uri = st.text_input(
            "Redirect URI (must match Connected App)",
            value="https://data-quality-ui-z6jhwctbwsffhaus82zxqm.streamlit.app/",
            key="sf_redirect_uri",
        )

        oauth_domain = st.selectbox(
            "Auth Domain", ["login", "test"], index=0, key="oauth_domain"
        )

        st.write("1) Click below to open the Salesforce consent page in a new tab/window.")

        auth_url: Optional[str] = None

        # Build PKCE challenge and URL when we have client_id+redirect
        if client_id and redirect_uri:
            verifier = st.session_state["pkce"].get(client_id)
            if not verifier:
                # First time for this client_id → generate verifier + challenge
                verifier = generate_code_verifier()
                st.session_state["pkce"][client_id] = verifier
                challenge = code_challenge_from_verifier(verifier)
                auth_url = build_salesforce_auth_url(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    domain=oauth_domain,
                    scopes="api refresh_token",
                    code_challenge=challenge,
                )
                st.session_state["pkce_auth_url"][client_id] = auth_url
            else:
                # Re-use existing URL so PKCE stays consistent
                auth_url = st.session_state["pkce_auth_url"].get(client_id)

        if auth_url:
            st.write("Authorization URL (open in browser):")
            st.code(auth_url)

        if st.button("Open Auth URL (copy into browser)"):
            if not auth_url:
                st.error("Provide Client ID and Redirect URI first.")
            else:
                st.info(
                    "Open the URL (copy/paste). After consenting, Salesforce will redirect "
                    "to your redirect URI with ?code=...."
                )

        st.markdown(
            "2) After consenting, copy the `code` parameter from the redirected URL and "
            "paste below to exchange it for tokens."
        )
        auth_code = st.text_input(
            "Paste authorization code (value of code query param)", key="oauth_code"
        )

        if st.button("Exchange code for tokens", key="exchange_code_btn"):
            if not (client_id and redirect_uri and auth_code):
                st.error("Provide Client ID, Redirect URI and the auth code.")
            else:
                try:
                    with st.spinner("Exchanging code for tokens..."):
                        code_verifier = st.session_state["pkce"].get(client_id)
                        if not code_verifier:
                            st.error(
                                "Missing PKCE verifier in session. "
                                "Re-start auth flow (re-generate auth url)."
                            )
                        else:
                            resp = exchange_code_for_token(
                                client_id=client_id,
                                client_secret=client_secret or None,
                                code=auth_code,
                                redirect_uri=redirect_uri,
                                domain=oauth_domain,
                                code_verifier=code_verifier,
                            )
                            oauth_tokens = resp
                            st.success(
                                "Token exchange successful. "
                                "You can now use the tokens to instantiate API calls."
                            )
                            st.write("Example keys returned:")
                            st.json(
                                {
                                    k: v
                                    for k, v in resp.items()
                                    if k
                                    in (
                                        "access_token",
                                        "refresh_token",
                                        "instance_url",
                                        "id",
                                    )
                                }
                            )

                            # create simple_salesforce instance and fetch metadata samples
                            try:
                                sf = sf_from_tokens(
                                    instance_url=resp["instance_url"],
                                    access_token=resp["access_token"],
                                )
                                st.success(
                                    "Connected to Salesforce via OAuth — "
                                    "simple_salesforce instance ready."
                                )
                                sobjects = fetch_org_sobjects(sf, limit=500)
                                sbqqs = [
                                    s for s in sobjects if s.upper().startswith("SBQQ__")
                                ]
                                st.write(
                                    "Top SBQQ objects (if present):", sbqqs[:20]
                                )
                                sel = st.multiselect(
                                    "Select sObjects to fetch sample rows (OAuth)",
                                    options=sobjects[:200],
                                    default=sbqqs[:4],
                                )
                                if sel:
                                    meta_folder = os.path.join(
                                        out_dir, "sf_oauth_fetch"
                                    )
                                    os.makedirs(meta_folder, exist_ok=True)
                                    for api_name in sel:
                                        try:
                                            df = fetch_sample_records(
                                                sf, api_name, limit=200
                                            )
                                            key = f"sf__{normalize_sf_name(api_name)}"
                                            sf_tables[key] = df
                                            csv_path = os.path.join(
                                                meta_folder,
                                                f"{normalize_sf_name(api_name)}__sample.csv",
                                            )
                                            df.to_csv(csv_path, index=False)
                                            saved_files.append(csv_path)
                                        except Exception as e:
                                            st.warning(
                                                f"Failed to fetch {api_name}: {e}"
                                            )
                                    st.success(
                                        f"Fetched {len(sf_tables)} objects; "
                                        f"saved to {meta_folder}"
                                    )
                            except Exception as e:
                                st.warning(
                                    "Tokens exchanged but failed to instantiate "
                                    f"API client: {e}"
                                )

                except Exception as e:
                    st.error("Code exchange failed: " + str(e))
                    with st.expander("Show full traceback"):
                        st.text(traceback.format_exc())

    # (Right column reserved for quick username+token connect later, if needed)
    return sf_tables, saved_files, oauth_tokens


# -------------------- Login card (non-blocking) --------------------


def login_card():
    st.markdown(
        """
        <style>
        .login-box { max-width:880px; margin:24px auto; }
        .login-title { font-size:22px; font-weight:700; margin-bottom:6px; }
        .login-sub { color:#666; margin-bottom:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-title">🔐 CPQ → RCA Analyzer by Jay</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="login-sub">Please sign in to continue. '
            "Create an account if you are new.</div>",
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["Login", "Create account"])

        with tab1:
            user_in = st.text_input("Username or Email", key="login_user")
            pwd_in = st.text_input(
                "Password", type="password", key="login_pwd"
            )
            if st.button("Login", key="login_btn", use_container_width=True):
                ok, username = auth_db.verify_user(user_in.strip(), pwd_in)
                if ok:
                    st.session_state["user"] = username
                    st.session_state["page"] = "upload"
                    st.rerun()
                else:
                    st.error("Invalid username/email or password")

        with tab2:
            new_user = st.text_input("Choose Username", key="signup_user")
            new_email = st.text_input("Email", key="signup_email")
            new_pwd = st.text_input(
                "Password", type="password", key="signup_pwd"
            )
            new_pwd2 = st.text_input(
                "Confirm Password", type="password", key="signup_pwd2"
            )
            if st.button(
                "Create Account", key="signup_btn", use_container_width=True
            ):
                if new_pwd != new_pwd2:
                    st.error("Passwords do not match")
                elif not new_user or not new_email or not new_pwd:
                    st.error("All fields are required")
                else:
                    ok, msg = auth_db.create_user(
                        new_user.strip(), new_email.strip(), new_pwd
                    )
                    if ok:
                        st.success("Account created — please login.")
                    else:
                        st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Analysis-section rendering helpers --------------------


def render_pk_section(report: Dict[str, Any]):
    st.subheader("1) Primary Key uniqueness issues")
    pk_issues = report.get("pk_uniqueness_issues", {})
    pk_rows = []
    for tname, info in pk_issues.items():
        pk_col = info.get("pk_col")
        for pk_val, details in info.get("dup_values", {}).items():
            pk_rows.append(
                {
                    "file": tname,
                    "pk_column": pk_col,
                    "pk_value": pk_val,
                    "duplicate_count": details.get("count", 0),
                    "sample_preview": sample_preview(
                        details.get("examples", [])
                    ),
                }
            )
    if pk_rows:
        st.dataframe(
            pd.DataFrame(pk_rows).sort_values(
                ["file", "duplicate_count"], ascending=[True, False]
            ),
            height=240,
        )
    else:
        st.write("No primary-key duplicates detected.")


def render_full_row_section(report: Dict[str, Any]):
    st.subheader("2) Full-row duplicates")
    full_row = report.get("full_row_duplicates", {})
    fr_rows = []
    for tname, info in full_row.items():
        groups = info.get("groups", {})
        for sig, g in groups.items():
            fr_rows.append(
                {
                    "file": tname,
                    "signature": sig,
                    "duplicate_count": g.get("count", 0),
                    "sample_preview": sample_preview(g.get("examples", [])),
                }
            )
    if fr_rows:
        st.dataframe(
            pd.DataFrame(fr_rows).sort_values(
                ["file", "duplicate_count"], ascending=[True, False]
            ),
            height=240,
        )
    else:
        st.write("No full-row duplicates detected.")


def render_logical_business_section(report: Dict[str, Any]):
    st.subheader("3) Logical / Business Key Uniqueness")

    parent_map = report.get("parent_merge_map", {})
    if parent_map:
        st.markdown("**Logical duplicate parent merges (canonical mapping):**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"duplicate_pk": k, "canonical": v}
                    for k, v in parent_map.items()
                ]
            ),
            height=200,
        )
    else:
        st.write("No logical duplicate parent merges found.")

    bk_issues = report.get("business_key_issues", {})
    if bk_issues:
        st.markdown("**Business key duplicate issues:**")
        bk_rows = []
        for t, issues in bk_issues.items():
            for i in issues:
                bk_rows.append(
                    {
                        "file": t,
                        "business_key": "|".join(i.get("key", []))
                        if isinstance(i.get("key", []), list)
                        else str(i.get("key")),
                        "dup_count": i.get("dup_count", 0),
                    }
                )
        st.dataframe(pd.DataFrame(bk_rows), height=200)
    else:
        st.write("No business key duplicate issues detected.")


def render_fk_consistency_section(report: Dict[str, Any]):
    st.subheader("4) FK Consistency & 1:1 Integrity")

    # FK merge conflicts
    fk_conf = report.get("fk_merge_conflicts", {})
    if fk_conf.get("conflicts"):
        st.markdown("**FK merge conflicts (suggested remapping):**")
        rows = []
        for c in fk_conf["conflicts"]:
            rows.append(
                {
                    "child_table": c.get("child_table"),
                    "fk_column": c.get("fk_column"),
                    "canonical": c.get("canonical"),
                    "rows_affected": c.get("rows_affected"),
                    "suggested_sql": (c.get("suggested_updates") or [""])[0],
                }
            )
        st.dataframe(pd.DataFrame(rows), height=240)
    else:
        st.write("No FK merge conflicts detected.")

    st.markdown("---")
    # 1-to-1 integrity issues
    o2o = report.get("one_to_one_issues", [])
    st.markdown("**1-to-1 integrity violations (if any):**")
    if o2o:
        st.dataframe(pd.DataFrame(o2o), height=220)
    else:
        st.write("No 1:1 integrity issues detected.")

    st.markdown("---")
    # Deduplication re-linking suggestions
    st.markdown("**Deduplication re-linking (child FK conflicts):**")
    cfc = report.get("child_fk_conflicts", [])
    if cfc:
        st.dataframe(pd.DataFrame(cfc), height=220)
    else:
        st.write("No child FK re-linking suggestions found.")


def render_referential_section(report: Dict[str, Any]):
    st.subheader("5) Referential Integrity — Orphans")
    orphs = report.get("expanded_orphans", [])
    if orphs:
        orrows = []
        for o in orphs:
            fkcol = (
                o.get("fk")
                if isinstance(o.get("fk"), str)
                else ",".join(o.get("fk") or [])
            )
            orrows.append(
                {
                    "child": o.get("from"),
                    "parent": o.get("to"),
                    "fk": fkcol,
                    "missing_count": o.get("count", 0),
                }
            )
        st.dataframe(pd.DataFrame(orrows), height=220)
    else:
        st.write("No orphan records detected.")


def render_remediation_section(report: Dict[str, Any], out_dir: str):
    st.markdown("---")
    st.subheader("Remediation Artifacts")

    rem = report.get("remediation_artifacts", {}) or {}
    fk_updates_path = rem.get("fk_updates_sql")
    child_rows_path = rem.get("child_rows_to_fix_csv")

    if fk_updates_path and os.path.exists(fk_updates_path):
        with open(fk_updates_path, "rb") as f:
            st.download_button(
                "Download fk_updates.sql",
                f,
                file_name=os.path.basename(fk_updates_path),
            )

    if child_rows_path and os.path.exists(child_rows_path):
        with open(child_rows_path, "rb") as f:
            st.download_button(
                "Download child_rows_to_fix.csv",
                f,
                file_name=os.path.basename(child_rows_path),
            )

    st.download_button(
        "Download full JSON report",
        json.dumps(report, indent=2),
        file_name=(
            f"analysis_report_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        ),
        mime="application/json",
    )


# -------------------- Upload & Analysis UI --------------------


def upload_page():
    """First page: upload, configure, then jump to analysis."""

    st.title(f"CPQ → RCA Analyzer (User: {st.session_state.get('user')})")
    if st.button("Logout"):
        st.session_state["user"] = None
        st.session_state["page"] = "upload"
        st.rerun()

    st.markdown("---")

    # show reference PDF path and download button if present
    if os.path.exists(SAMPLE_PDF_PATH):
        st.markdown(f"📄 Reference PDF available at: `{SAMPLE_PDF_PATH}`")
        try:
            with open(SAMPLE_PDF_PATH, "rb") as pf:
                st.download_button(
                    "Download reference PDF",
                    pf,
                    file_name=os.path.basename(SAMPLE_PDF_PATH),
                )
        except Exception:
            pass

    st.markdown(
        "Upload CPQ CSV/Excel exports (Product, Pricebook, Quote, "
        "QuoteLine, etc.)."
    )

    uploaded_files = st.file_uploader(
        "Upload files (CSV/XLSX)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
    )

    bkmap_file = st.file_uploader(
        "Optional: Business Key Map (JSON/YAML)",
        type=["json", "yml", "yaml"],
        accept_multiple_files=False,
    )

    out_dir = st.text_input("Output folder", value="analysis_output")

    if "last_out_dir" not in st.session_state:
        st.session_state["last_out_dir"] = out_dir
    st.session_state["last_out_dir"] = out_dir

    # Salesforce connect (OAuth)
    sf_tables, sf_saved, oauth_tokens = salesforce_connect_panel(out_dir)

    # read uploaded files into tables
    tables: Dict[str, pd.DataFrame] = {}
    if uploaded_files:
        for f in uploaded_files:
            name = os.path.splitext(f.name)[0]
            try:
                content = f.read()
                # try as bytes, stream, or saved file
                try:
                    obj = analyzer.read_table(content)
                except Exception:
                    try:
                        f.seek(0)
                        obj = analyzer.read_table(f)
                    except Exception:
                        tmp = os.path.join(out_dir, f.name)
                        with open(tmp, "wb") as wf:
                            wf.write(content)
                        obj = analyzer.read_table(tmp)
                if isinstance(obj, dict):
                    for sheet, df in obj.items():
                        tables[f"{name}__{sheet}"] = df
                else:
                    tables[name] = obj
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")

    # Optional: show OAuth token keys (masked)
    if oauth_tokens:
        st.success(
            "OAuth tokens are available in-memory for this session "
            "(not saved to disk)."
        )
        if st.checkbox("Show returned token keys (masked)"):
            masked = {
                k: (v[:8] + "..." if isinstance(v, str) and len(v) > 8 else v)
                for k, v in oauth_tokens.items()
            }
            st.json(masked)

    # business key map
    business_key_map = {}
    if bkmap_file:
        try:
            bk_bytes = bkmap_file.read()
            bk_path = os.path.join(out_dir, bkmap_file.name)
            os.makedirs(out_dir, exist_ok=True)
            with open(bk_path, "wb") as wf:
                wf.write(bk_bytes)
            business_key_map = analyzer.load_business_key_map(bk_path)
            st.success("Business key map loaded.")
        except Exception as e:
            st.warning(f"Could not load business key map: {e}")

    # Load local sample files (env)
    if st.button("Load local sample files (env)"):
        sample_paths = [
            "/mnt/data/Products Data Table.xlsx",
            "/mnt/data/Product Prices Data Table.xlsx",
            "/mnt/data/Orders Data Table.xlsx",
        ]
        tables_local = {}
        for p in sample_paths:
            if os.path.exists(p):
                try:
                    obj = analyzer.read_table(p)
                    if isinstance(obj, dict):
                        for sheet, df in obj.items():
                            tables_local[
                                f"{os.path.splitext(os.path.basename(p))[0]}__{sheet}"
                            ] = df
                    else:
                        tables_local[
                            os.path.splitext(os.path.basename(p))[0]
                        ] = obj
                except Exception as e:
                    st.warning(f"Failed to read {p}: {e}")
            else:
                st.warning(f"Sample file not found: {p}")

        if tables_local:
            st.success(f"Loaded {len(tables_local)} tables from sample files.")
            merged = {}
            merged.update(tables_local)
            merged.update(sf_tables or {})
            if not merged:
                st.error(
                    "No tables to analyze from samples or Salesforce."
                )
            else:
                os.makedirs(out_dir, exist_ok=True)
                with st.spinner("Running analyzer on sample files..."):
                    report = analyzer.generate_analysis_report(
                        merged,
                        business_key_map=business_key_map,
                        output_folder=out_dir,
                    )
                st.session_state["cached_report"] = report
                st.session_state["cached_out_dir"] = out_dir
                st.session_state["page"] = "analysis"
                st.rerun()

    st.markdown("---")
    st.markdown(
        "Once files are uploaded and (optionally) Salesforce data is fetched, "
        "click below to go to the analysis page."
    )

    if st.button("Click here for the Analysis"):
        merged = {}
        merged.update(tables)
        merged.update(sf_tables or {})

        if not merged:
            st.error(
                "No tables to analyze — upload files or fetch from Salesforce."
            )
            return

        os.makedirs(out_dir, exist_ok=True)
        with st.spinner("Running analyzer..."):
            report = analyzer.generate_analysis_report(
                merged,
                business_key_map=business_key_map,
                output_folder=out_dir,
            )

        st.session_state["cached_report"] = report
        st.session_state["cached_out_dir"] = out_dir
        st.session_state["page"] = "analysis"
        st.rerun()


def analysis_page():
    """Second page: show 5 independent analysis sections using cached report."""

    st.title(f"CPQ → RCA Analyzer (User: {st.session_state.get('user')})")
    top_cols = st.columns([1, 3])
    with top_cols[0]:
        if st.button("Back to Upload Page"):
            st.session_state["page"] = "upload"
            st.rerun()
    with top_cols[1]:
        if st.button("Logout"):
            st.session_state["user"] = None
            st.session_state["page"] = "upload"
            st.rerun()

    st.markdown("---")

    report = st.session_state.get("cached_report")
    out_dir = st.session_state.get("cached_out_dir") or "analysis_output"

    if not report:
        st.error(
            "No analysis report in session. Please go back to the upload "
            "page and run the analysis first."
        )
        return

    # ----- Summary header -----
    st.header("Analysis Summary")
    validation = report.get("validation", {})
    st.write(f"Tables uploaded: **{validation.get('uploaded_count', 0)}**")
    st.write(f"Tables analyzed: **{validation.get('validated_count', 0)}**")
    st.write(
        f"Inferred relationships: **{len(report.get('relationships', []))}**"
    )
    st.write(
        f"Orphan issues (expanded): **{len(report.get('expanded_orphans', []))}**"
    )

    st.markdown("---")

    # -------- Section 1: PK uniqueness --------
    st.markdown("### Section 1: Primary Key Uniqueness")
    if st.button("Run Analysis", key="run_pk"):
        st.session_state["show_pk_section"] = True
    if st.session_state.get("show_pk_section"):
        render_pk_section(report)

    st.markdown("---")

    # -------- Section 2: Full-row duplicates --------
    st.markdown("### Section 2: Full-row Duplicates")
    if st.button("Run Analysis", key="run_fr"):
        st.session_state["show_fr_section"] = True
    if st.session_state.get("show_fr_section"):
        render_full_row_section(report)

    st.markdown("---")

    # -------- Section 3: Logical / Business keys --------
    st.markdown("### Section 3: Logical / Business Key Uniqueness")
    if st.button("Run Analysis", key="run_logical"):
        st.session_state["show_logical_section"] = True
    if st.session_state.get("show_logical_section"):
        render_logical_business_section(report)

    st.markdown("---")

    # -------- Section 4: FK consistency --------
    st.markdown("### Section 4: FK Consistency & 1:1 Integrity")
    if st.button("Run Analysis", key="run_fk"):
        st.session_state["show_fk_section"] = True
    if st.session_state.get("show_fk_section"):
        render_fk_consistency_section(report)

    st.markdown("---")

    # -------- Section 5: Referential integrity --------
    st.markdown("### Section 5: Referential Integrity (Orphans)")
    if st.button("Run Analysis", key="run_ref"):
        st.session_state["show_ref_section"] = True
    if st.session_state.get("show_ref_section"):
        render_referential_section(report)

    # -------- Remediation artifacts --------
    render_remediation_section(report, out_dir)


# -------------------- Analyzer UI entry --------------------


def analyzer_ui():
    # decide page
    if "page" not in st.session_state:
        st.session_state["page"] = "upload"

    if st.session_state["page"] == "upload":
        upload_page()
    else:
        analysis_page()


# -------------------- Main --------------------


def main():
    auth_db.init_db()
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state.get("user") is None:
        login_card()
    else:
        analyzer_ui()


if __name__ == "__main__":
    main()
