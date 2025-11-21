# -------------------- UI: Salesforce connection panel (PKCE) --------------------
def salesforce_connect_panel(out_dir: str = "analysis_output") -> Tuple[Dict[str, pd.DataFrame], List[str], Optional[dict]]:
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

    # Ensure pkce storage exists
    if "pkce" not in st.session_state:
        st.session_state["pkce"] = {}
    if "pkce_auth_url" not in st.session_state:
        st.session_state["pkce_auth_url"] = {}

    with col1:
        st.markdown("#### OAuth / Connected App (PKCE)")
        st.markdown("Create a Connected App in Salesforce and supply Client ID/Secret + redirect URI.")
        st.caption("Scopes: `api` and `refresh_token`. For local dev you can use redirect `http://localhost:8501/`, for Cloud use your deployed URL.")

        client_id = st.text_input("Connected App Client ID (Consumer Key)", key="sf_client_id")
        client_secret = st.text_input("Connected App Client Secret (optional)", type="password", key="sf_client_secret")
        # default to your deployed Streamlit Cloud redirect URI — ensure this EXACT string is configured in the Connected App
        redirect_uri = st.text_input(
            "Redirect URI (must match Connected App)",
            value="https://data-quality-ui-z6jhwctbwsffhaus82zxqm.streamlit.app/",
            key="sf_redirect_uri",
        )
        oauth_domain = st.selectbox("Auth Domain", ["login", "test"], index=0, key="oauth_domain")

        # ------------------------------------------------------------------
        # 🔥 IMPORTANT: reset PKCE state when Client ID changes
        # ------------------------------------------------------------------
        prev_client_id = st.session_state.get("prev_client_id")
        if prev_client_id and prev_client_id != client_id:
            # throw away old verifier + auth URL for previous client_id
            st.session_state["pkce"].pop(prev_client_id, None)
            st.session_state["pkce_auth_url"].pop(prev_client_id, None)
        st.session_state["prev_client_id"] = client_id
        # ------------------------------------------------------------------

        st.write("1) Click below to open the Salesforce consent page in a new tab/window.")
        auth_url = None

        # Build PKCE challenge and URL when we have client_id+redirect
        if client_id and redirect_uri:
            # only generate verifier/challenge once per client_id,
            # otherwise Streamlit reruns will break PKCE
            if client_id not in st.session_state["pkce"]:
                verifier = generate_code_verifier()
                challenge = code_challenge_from_verifier(verifier)
                st.session_state["pkce"][client_id] = verifier

                auth_url = build_salesforce_auth_url(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    domain=oauth_domain,
                    scopes="api refresh_token",
                    code_challenge=challenge,
                )
                # store URL so we don’t rebuild with a new challenge
                st.session_state["pkce_auth_url"][client_id] = auth_url
            else:
                # reuse existing auth URL
                auth_url = st.session_state["pkce_auth_url"].get(client_id)

            if auth_url:
                st.write("Authorization URL (open in browser):")
                st.code(auth_url)

        if st.button("Open Auth URL (copy into browser)"):
            if not auth_url:
                st.error("Provide Client ID and Redirect URI first.")
            else:
                st.info(
                    "Open the URL (copy/paste if browser didn't open). After consenting, "
                    "Salesforce will redirect to your redirect URI with `?code=...`."
                )

        st.markdown(
            "2) After consenting, copy the `code` parameter from the redirected URL "
            "and paste below to exchange it for tokens."
        )
        auth_code = st.text_input("Paste authorization code (value of `code` query param)", key="oauth_code")

        if st.button("Exchange code for tokens", key="exchange_code_btn"):
            if not (client_id and redirect_uri and auth_code):
                st.error("Provide Client ID, Redirect URI and the auth code.")
            else:
                try:
                    with st.spinner("Exchanging code for tokens..."):
                        code_verifier = st.session_state["pkce"].get(client_id)
                        if not code_verifier:
                            st.error("Missing PKCE verifier in session. Re-start auth flow (re-generate auth url).")
                        else:
                            # Exchange code for tokens (PKCE-aware)
                            resp = exchange_code_for_token(
                                client_id=client_id,
                                client_secret=client_secret or None,
                                code=auth_code,
                                redirect_uri=redirect_uri,
                                domain=oauth_domain,
                                code_verifier=code_verifier,
                            )
                            oauth_tokens = resp
                            st.success("Token exchange successful. You can now use the tokens to instantiate API calls.")
                            st.write("Example keys returned:")
                            st.json(
                                {
                                    k: v
                                    for k, v in resp.items()
                                    if k in ("access_token", "refresh_token", "instance_url", "id")
                                }
                            )

                            # create simple_salesforce instance and fetch metadata samples
                            try:
                                sf = sf_from_tokens(instance_url=resp["instance_url"], access_token=resp["access_token"])
                                st.success("Connected to Salesforce via OAuth — simple_salesforce instance ready.")
                                # optionally fetch SBQQ objects
                                sobjects = fetch_org_sobjects(sf, limit=500)
                                sbqqs = [s for s in sobjects if s.upper().startswith("SBQQ__")]
                                st.write("Top SBQQ objects (if present):", sbqqs[:20])
                                sel = st.multiselect(
                                    "Select sObjects to fetch sample rows (OAuth)",
                                    options=sobjects[:200],
                                    default=sbqqs[:4],
                                )
                                if sel:
                                    meta_folder = os.path.join(out_dir, "sf_oauth_fetch")
                                    os.makedirs(meta_folder, exist_ok=True)
                                    for api_name in sel:
                                        try:
                                            df = fetch_sample_records(sf, api_name, limit=200)
                                            key = f"sf__{normalize_sf_name(api_name)}"
                                            sf_tables[key] = df
                                            csv_path = os.path.join(
                                                meta_folder, f"{normalize_sf_name(api_name)}__sample.csv"
                                            )
                                            df.to_csv(csv_path, index=False)
                                            saved_files.append(csv_path)
                                        except Exception as e:
                                            st.warning(f"Failed to fetch {api_name}: {e}")
                                    st.success(f"Fetched {len(sf_tables)} objects; saved to {meta_folder}")
                            except Exception as e:
                                st.warning("Tokens exchanged but failed to instantiate API client: " + str(e))
                except Exception as e:
                    st.error("Code exchange failed: " + str(e))
                    with st.expander("Show full traceback"):
                        st.text(traceback.format_exc())

    # (You can keep your quick username+token column here later if you want.)

    return sf_tables, saved_files, oauth_tokens
