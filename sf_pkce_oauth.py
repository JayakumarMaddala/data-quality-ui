# sf_pkce_oauth.py
# PKCE helpers + build/exchange functions for Salesforce OAuth2
import base64
import hashlib
import secrets
from urllib.parse import urlencode
import requests

# ------- PKCE helpers -------
def generate_code_verifier(length: int = 64) -> str:
    """Generate a URL-safe code_verifier"""
    return base64.urlsafe_b64encode(secrets.token_bytes(length)).rstrip(b"=").decode("ascii")

def code_challenge_from_verifier(verifier: str) -> str:
    """Return base64url-encoded SHA256 of verifier"""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

# ------- Build authorization URL (PKCE) -------
def build_salesforce_auth_url(client_id: str, redirect_uri: str, domain: str = "login",
                              scopes: str = "api refresh_token", state: str | None = None,
                              code_challenge: str | None = None):
    """
    Returns: (auth_url, code_verifier) — the caller should persist code_verifier in session
    """
    if code_challenge is None:
        raise ValueError("code_challenge must be provided")
    auth_base = f"https://{domain}.salesforce.com/services/oauth2/authorize"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if state:
        params["state"] = state
    return auth_base + "?" + urlencode(params)

# ------- Exchange authorization code for tokens (include code_verifier) -------
def exchange_code_for_token(client_id: str, client_secret: str | None, code: str,
                            redirect_uri: str, domain: str, code_verifier: str, timeout: int = 30):
    token_url = f"https://{domain}.salesforce.com/services/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    # include client_secret only if provided / required
    if client_secret:
        data["client_secret"] = client_secret

    resp = requests.post(token_url, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# ------- Helper to create a simple_salesforce instance from tokens -------
def sf_from_tokens(instance_url: str, access_token: str):
    """
    Returns a minimal simple_salesforce instance using the access token.
    Note: simple_salesforce expects session_id and instance_url.
    """
    from simple_salesforce import Salesforce
    return Salesforce(instance_url=instance_url, session_id=access_token)
