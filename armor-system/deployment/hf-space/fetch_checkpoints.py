"""Download the Model 1 v1 checkpoints from a private Hugging Face repo.

The Space itself is public so the ISAs can open the URL, but the trained weights
are not published: they live in a private model repo and are pulled at start-up
with the token stored as the Space secret HF_TOKEN.

Nothing here verifies the weights. It does not need to. `service.py` hashes every
checkpoint against the frozen test evidence when it loads and refuses to start if
any file is missing or altered, so a bad download fails loudly a second later.

Environment:
    HF_TOKEN              a read token for the private repo (Space secret)
    MODEL1_HF_REPO        e.g. "your-name/armor-model1-checkpoints"
    MODEL1_CHECKPOINT_ROOT where to put them; must match what server.py reads
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = os.environ.get("MODEL1_HF_REPO", "").strip()
DEST = os.environ.get("MODEL1_CHECKPOINT_ROOT", "/home/user/checkpoints")
TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

# The ten files service.py requires. Narrowing the download keeps a stray large
# file in that repo from being pulled into a free Space on every cold start.
PATTERNS = [f"model1_v1/seed_{s}/s2_best.{ext}" for s in range(5) for ext in ("pt", "json")]


def main():
    if not REPO:
        sys.exit("MODEL1_HF_REPO is not set. Set it in the Space variables to the "
                 "private repo holding model1_v1/seed_{0..4}/s2_best.{pt,json}.")
    if TOKEN is None:
        print("warning: HF_TOKEN is not set; this only works if the repo is public",
              file=sys.stderr)

    os.makedirs(DEST, exist_ok=True)
    print(f"downloading {len(PATTERNS)} checkpoint files from {REPO} into {DEST}")
    snapshot_download(repo_id=REPO, repo_type="model", token=TOKEN,
                      local_dir=DEST, allow_patterns=PATTERNS)

    missing = [p for p in PATTERNS if not os.path.isfile(os.path.join(DEST, p))]
    if missing:
        sys.exit("these files were not in the repo:\n  " + "\n  ".join(missing))
    print("all ten checkpoint files present; service.py will now hash them")


if __name__ == "__main__":
    main()
