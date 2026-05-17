"""
generate_halo_http_key.py
=========================
Generates a shared HMAC-SHA256 key for authenticating HTTP traffic
between Python HALO agents and ESP32 devices.

Run once:
    python generate_halo_http_key.py

Outputs:
  - .keys/halo_http_key.hex   (for Python — loaded by base_agent)
  - Instructions for embedding the key in your ESP32 firmware

The same key must be in both places. Keep it secret — treat it like
a password. Don't commit it to git.
"""

import os
import secrets

KEY_DIR  = ".keys"
KEY_FILE = os.path.join(KEY_DIR, "halo_http_key.hex")

os.makedirs(KEY_DIR, exist_ok=True)

key_bytes = secrets.token_bytes(32)   # 256-bit key
key_hex   = key_bytes.hex()

with open(KEY_FILE, "w") as f:
    f.write(key_hex)

try:
    os.chmod(KEY_FILE, 0o600)
except Exception:
    pass

# Format as C array for copy-pasting into Arduino sketch
c_array = ", ".join(f"0x{key_bytes[i]:02x}" for i in range(32))

print(f"\n✅  Key written to {KEY_FILE}")
print(f"\n── Python env var (add to docker-compose or .env) ──────────────")
print(f"HALO_HTTP_KEY={key_hex}")
print(f"\n── Arduino firmware  (paste into halo_esp32_agent.ino) ─────────")
print(f"const uint8_t HALO_HMAC_KEY[32] = {{{c_array}}};")
print(f"\n⚠️  Keep this key secret. Same value must be in both places.\n")