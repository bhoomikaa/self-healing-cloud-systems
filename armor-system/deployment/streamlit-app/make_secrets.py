"""Turn a Google service-account JSON key into the app's Streamlit secrets block.

Streamlit wants TOML, Google hands you JSON, and the private key is the part
that goes wrong: it is one long line with literal \\n escapes, and hand-editing
it into a TOML file usually either breaks the newlines or wraps the value in the
wrong kind of quotes. This does the conversion so that does not happen.

    python -I make_secrets.py path/to/key.json --hf-repo bhoomikaa7/armor-model1-checkpoints

Prints the block to paste into the app's Settings -> Secrets. Add your Hugging
Face read token to the HF_TOKEN line afterwards.

Nothing is written to disk and nothing is sent anywhere. The key file itself
should never be committed: it is a credential, this repository is public, and a
leaked service-account key is a live door into the project until it is revoked.
Delete it from your downloads once the secrets are saved; the app holds the only
copy it needs.
"""
from __future__ import annotations

import argparse
import json
import sys

# Order matters only for readability; Streamlit reads it as a mapping.
FIELDS = ("type", "project_id", "private_key_id", "private_key", "client_email",
          "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url",
          "client_x509_cert_url", "universe_domain")


def toml_escape(value: str) -> str:
    """Escape a string for a TOML basic (double-quoted) string.

    The private key arrives from Google already containing the two characters
    backslash and n where newlines belong. A TOML basic string turns \\n back
    into a real newline, which is exactly what the credentials parser needs, so
    those sequences are left alone and only quotes and stray backslashes are
    escaped.
    """
    out = []
    i = 0
    while i < len(value):
        c = value[i]
        if c == "\\" and i + 1 < len(value) and value[i + 1] in "n\"\\":
            out.append(value[i:i + 2])
            i += 2
            continue
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("key_file", help="the service-account JSON downloaded from Google Cloud")
    ap.add_argument("--hf-repo", required=True, help="private Hugging Face repo holding the checkpoints")
    a = ap.parse_args(argv)

    with open(a.key_file, encoding="utf-8") as fh:
        key = json.load(fh)

    if key.get("type") != "service_account":
        sys.exit(f"{a.key_file} is not a service-account key (type={key.get('type')!r}).")
    missing = [f for f in ("project_id", "private_key", "client_email") if not key.get(f)]
    if missing:
        sys.exit(f"the key file is missing: {', '.join(missing)}")

    print("# ---- paste everything below into Settings -> Secrets ----")
    print(f'MODEL1_HF_REPO = "{a.hf_repo}"')
    print('HF_TOKEN = "PUT_YOUR_HUGGING_FACE_READ_TOKEN_HERE"')
    print()
    print("[gcp_service_account]")
    for f in FIELDS:
        if f in key:
            print(f'{f} = "{toml_escape(str(key[f]))}"')
    print("# ---- end ----")
    print()
    print(f"service account: {key['client_email']}", file=sys.stderr)
    print(f"project:         {key['project_id']}", file=sys.stderr)
    print("Delete the downloaded key file once the secrets are saved.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
