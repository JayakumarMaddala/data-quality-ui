#!/usr/bin/env python3
"""
PKCE OAuth2 helper for Salesforce (Authorization Code + PKCE)

Saves no secrets. Prints access_token + refresh_token JSON to stdout.

Reference (local uploaded file):
file:///mnt/data/CPQ Book by Chandra.pdf
"""

import os
import base64
import hashlib
import secrets
import threading
import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import json
import time

# ======== CONFIGURE THESE ========
CLIENT_ID = "YOUR_CONNECTED_APP_CLIENT_ID"
# If your connected app requires client_secret for the web server flow, add it here.
# For PKCE public clients you typically DON'T send client_secret.
CLIENT_SECRET = None  # or "YOUR_CLIENT_SECRET" if required
REDIRECT_HOST = "localhost"
REDIRECT_PORT = 8501
REDIRECT_PATH = "/"
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}{REDIRECT_PATH}"
# Use login or test depending on your org:
SALESFORCE_DOMAIN = "login"   # use "test" for sandboxes
# Scopes you need:
SCOPE = "api refresh_token"
# ================================

AUTH_URL = f"https://{SALESFORCE_DOMAIN}.salesforce.com/services/oauth2/authorize"
TOKEN_URL = f"https://{SALESFORCE_DOMAIN}.salesforce.com/services/oauth2/token"

# Globals to pass code from handler to main thread
_received_auth_code = None
_received_error = None

def generate_code_verifier(length: int = 64) -> str:
    # high-entropy random string (43..128). We'll produce 64 bytes -> base64url -> ~86 chars
    verifier = base64.urlsafe_b64encode(os.urandom(length)).rstrip(b"=").decode("utf-8")
    # Ensure length between 43 and 128
    if len(verifier) < 43:
        verifier += "A" * (43 - len(verifier))
    return verifier

def generate_code_challenge(verifier: str) -> str:
    sha256 = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(sha256).rstrip(b"=").decode("utf-8")
    return challenge

class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _received_auth_code, _received_error
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        # simple response page
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        if "error" in qs:
            _received_error = qs.get("error_description", qs.get("error", ["unknown error"]))[0]
            self.wfile.write(b"<html><body><h2>OAuth Error</h2><pre>")
            self.wfile.write(str(_received_error).encode("utf-8"))
            self.wfile.write(b"</pre><p>Close this window and check your app.</p></body></html>")
            return

        code = qs.get("code", [None])[0]
        if not code:
            self.wfile.write(b"<html><body><h2>No code received</h2><p>Check the redirect params.</p></body></html>")
            return

        # store the code for main thread
        _received_auth_code = code
        self.wfile.write(b"<html><body><h2>Authorization received</h2><p>You can close this window.</p></body></html>")

    def log_message(self, format, *args):
        # silence default logging (optional)
        return

def start_http_server(server_class=HTTPServer, handler_class=OAuthHandler):
    server_address = (REDIRECT_HOST, REDIRECT_PORT)
    httpd = server_class(server_address, handler_class)
    # run server until code received (we will shut it down from the main thread)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd

def build_authorize_url(client_id, redirect_uri, scope, code_challenge):
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        # optionally add "prompt=consent" to force showing the allow screen
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(params)

def exchange_code_for_token(code, client_id, redirect_uri, code_verifier, client_secret=None):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier
    }
    if client_secret:
        data["client_secret"] = client_secret

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(TOKEN_URL, data=data, headers=headers, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # surface server response for debugging
        raise RuntimeError(f"Token exchange failed: {resp.status_code} {resp.text}") from e
    return resp.json()

def main():
    global _received_auth_code, _received_error

    if CLIENT_ID.startswith("YOUR_"):
        print("ERROR: edit the script and provide your CLIENT_ID (Connected App).")
        return

    print(f"Starting local listener at {REDIRECT_URI} to capture the authorization code...")
    httpd = start_http_server()

    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    print(f"Generated code_verifier (kept secret) length={len(code_verifier)}")
    # DO NOT print the verifier in production logs - kept minimal for debug here.

    auth_url = build_authorize_url(CLIENT_ID, REDIRECT_URI, SCOPE, code_challenge)
    print("\nOpening browser for authorization. If it does not open, paste this URL into a browser:\n")
    print(auth_url + "\n")
    webbrowser.open(auth_url, new=2, autoraise=True)

    # wait for the authorization code (with a timeout)
    wait_seconds = 300
    poll_interval = 0.5
    waited = 0.0
    try:
        while waited < wait_seconds:
            if _received_error:
                raise RuntimeError(f"Authorization error received: {_received_error}")
            if _received_auth_code:
                break
            time.sleep(poll_interval)
            waited += poll_interval
        else:
            raise RuntimeError("Timed out waiting for authorization code (increase wait_seconds).")

        code = _received_auth_code
        print("Authorization code received. Exchanging for tokens...")

        token_response = exchange_code_for_token(code, CLIENT_ID, REDIRECT_URI, code_verifier, CLIENT_SECRET)
        print("\nToken response (JSON):")
        print(json.dumps(token_response, indent=2))

        # Optionally write tokens to file (BE CAREFUL with secrets); uncomment to enable:
        # with open("sf_tokens.json", "w") as f:
        #     json.dump(token_response, f)

    finally:
        # shutdown http server
        try:
            httpd.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
