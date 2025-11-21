# sf_test.py
from simple_salesforce import SalesforceLogin, Salesforce
import os, traceback

username = os.getenv("SF_USER") or "your.username@domain.com"
password = os.getenv("SF_PASS") or "your_password"
token = os.getenv("SF_TOKEN") or "your_security_token"
domain = os.getenv("SF_DOMAIN") or "login"  # or 'test' for sandbox

print("Attempting login to Salesforce...")
try:
    session_id, instance = SalesforceLogin(username=username, password=password, security_token=token, domain=domain)
    sf = Salesforce(instance=instance, session_id=session_id)
    print("SUCCESS: connected to instance:", instance)
    desc = sf.describe()
    print("sObjects found:", len(desc.get("sobjects", [])))
except Exception as e:
    print("Connection failed:", e)
    traceback.print_exc()
    # helpful hints
    print("\nHints:")
    print("- Ensure username/password works in the browser.")
    print("- If login via API fails but web login succeeds, try resetting security token.")
    print("- If error is INVALID_LOGIN, check user locked, API permission, domain and token handling.")