import asyncio
import aiohttp
import json
import os
import secrets
import hashlib
import base64
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

CLIENT_ID     = os.environ["MIELE_CLIENT_ID"]
CLIENT_SECRET = os.environ["MIELE_CLIENT_SECRET"]
REDIRECT_URI  = "http://localhost:8080/callback"
TOKEN_FILE    = "generic/miele_tokens.json"

AUTH_URL  = "https://api.mcs3.miele.com/thirdparty/login"
TOKEN_URL = "https://api.mcs3.miele.com/thirdparty/token"
# PKCE
code_verifier  = secrets.token_urlsafe(64)
code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier.encode()).digest()
).rstrip(b"=").decode()

received_code  = None
received_state = None
expected_state = secrets.token_urlsafe(16)


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global received_code, received_state
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        received_code  = params.get("code",  [None])[0]
        received_state = params.get("state", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Auth complete. You can close this tab.")

    def log_message(self, *args):
        pass  # suppress access logs


async def get_tokens(code: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            TOKEN_URL,
            data={
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  REDIRECT_URI,
                "client_id":     CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Token exchange failed ({resp.status}): {await resp.text()}")
            return await resp.json()


async def main():
    # Start local callback server
    server = HTTPServer(("localhost", 8080), CallbackHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Build auth URL
    params = urllib.parse.urlencode({
        "response_type":         "code",
        "client_id":             CLIENT_ID,
        "redirect_uri":          REDIRECT_URI,
        "scope":                 "openid email profile",
        "state":                 expected_state,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    })
    print(f"\nOpen this URL in your browser:\n\n{AUTH_URL}?{params}\n")
    print("Waiting for callback...")

    # Wait for callback
    while received_code is None:
        await asyncio.sleep(0.5)

    server.shutdown()

    if received_state != expected_state:
        raise RuntimeError("State mismatch — possible CSRF. Aborting.")

    tokens = await get_tokens(received_code)
    from datetime import datetime
    tokens["refreshed_at"] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)

    print(f"\nTokens saved to {TOKEN_FILE}")
    print(f"Access token expires in: {tokens.get('expires_in', '?')}s")
    print(f"Refresh token present: {'refresh_token' in tokens}")


if __name__ == "__main__":
    asyncio.run(main())
