"""Download CholecTrack20 from Synapse — credentials read from the environment.

Adapted from CAMMA's official downloader (Chinedu I. Nwoye) to keep all
secrets OUT of the file and out of version control. Set three env vars and
pass the destination folder:

    export SYNAPSE_EMAIL="you@example.com"
    export SYNAPSE_AUTH_TOKEN="<personal access token from Synapse>"
    export CHOLECTRACK20_ACCESS_KEY="<key from the CAMMA grant email>"
    python scripts/download_cholectrack20.py --dest ~/datasets/cholectrack20

The access key is personal (bound to the grantee's Synapse account); never
commit it, never share it. Per the CholecTrack20 DUA, code arising from
publications must be open — this env-var-based helper is safe to open-source
BECAUSE it contains no credentials.

Cite: Nwoye, Elgohary, Srinivas, Zaid, Lavanchy, Padoy. CholecTrack20:
A Multi-Perspective Tracking Dataset for Surgical Tools. CVPR 2025.
License: CC-BY-NC-SA 4.0.
"""

from __future__ import annotations

import argparse
import os
import sys


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"Missing required environment variable: {name} (see the "
                 "module docstring for the three that must be set).")
    return val


def main() -> None:
    ap = argparse.ArgumentParser(description="Download CholecTrack20 (Synapse)")
    ap.add_argument("--dest", required=True, help="local download folder")
    args = ap.parse_args()

    import requests
    import synapseclient
    import synapseutils

    email = require_env("SYNAPSE_EMAIL")
    auth_token = require_env("SYNAPSE_AUTH_TOKEN")
    access_key = require_env("CHOLECTRACK20_ACCESS_KEY")
    dest = os.path.abspath(os.path.expanduser(args.dest))
    os.makedirs(dest, exist_ok=True)

    print("Authenticating user ...", flush=True)
    syn = synapseclient.login(email=email, authToken=auth_token)

    # The validation endpoint is a free Render.com service that cold-starts:
    # the first request often times out just WAKING the dyno, and a retry then
    # hits a warm server. Retry with a generous timeout instead of failing.
    user_id = syn.getUserProfile()["ownerId"]
    payload = {"access_key": access_key, "synapse_id": user_id}
    entity_id = None
    for attempt in range(1, 6):
        print(f"Validating access key permission (attempt {attempt}/5, "
              "cold server may take a minute) ...", flush=True)
        try:
            resp = requests.post(
                "https://synapse-response.onrender.com/validate_access",
                json=payload, timeout=180,
            )
        except requests.exceptions.RequestException as e:
            print(f"  network hiccup ({type(e).__name__}); retrying ...", flush=True)
            continue
        if resp.status_code == 200:
            entity_id = resp.json()["entity_id"]
            break
        # A definitive rejection (bad key) should not be retried.
        if 400 <= resp.status_code < 500 and resp.status_code != 429:
            sys.exit(f"Access validation rejected ({resp.status_code}): {resp.text}")
        print(f"  server returned {resp.status_code}; retrying ...", flush=True)
    if entity_id is None:
        sys.exit("Access validation did not succeed after 5 attempts. The CAMMA "
                 "validation server (synapse-response.onrender.com) may be down; "
                 "wait a few minutes and re-run.")

    print(f"Downloading {entity_id} -> {dest} ...", flush=True)
    synapseutils.syncFromSynapse(syn, entity=entity_id, path=dest)
    print("success!", flush=True)


if __name__ == "__main__":
    main()
