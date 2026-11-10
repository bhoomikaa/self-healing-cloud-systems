#!/usr/bin/env bash
# Assemble a ready-to-push Hugging Face Space directory.
#
# The Space has to be self-contained: HF builds the image from the Space repo
# alone, so the armor-system tree is copied in rather than cloned at build time.
# That means it can drift from the repository, so this script records the commit
# it was built from in COMMIT.txt and the Space README links to it.
#
#   ./build_space.sh ~/armor-model1-space
#
# Then, from that directory:
#   git init && git remote add origin https://huggingface.co/spaces/<you>/<space>
#   git add -A && git commit -m "Deploy Model 1 v2 detection API" && git push -u origin main
set -euo pipefail

DEST="${1:?usage: build_space.sh <target-directory>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

echo "repository: $REPO"
echo "target:     $DEST"

mkdir -p "$DEST"
cp "$HERE/Dockerfile" "$HERE/README.md" "$HERE/entrypoint.sh" "$HERE/fetch_checkpoints.py" "$DEST/"

# Strip CR from the files that run inside the container. .gitattributes pins
# them to LF and the Dockerfile strips them again, but this is the step that
# actually crosses from a possibly-Windows checkout into the Space, so it is the
# right place to be sure. A CRLF shebang fails as a bare "not found".
for f in Dockerfile entrypoint.sh; do
  tr -d '' < "$DEST/$f" > "$DEST/$f.lf" && mv "$DEST/$f.lf" "$DEST/$f"
done
chmod +x "$DEST/entrypoint.sh"

# Same exclusions as ../model1-serving/Dockerfile.dockerignore: the 494 MB of
# raw and processed D1 data is never read by the service, and Airflow's log
# directory holds a symlink that breaks the copy on Windows.
rm -rf "$DEST/armor-system"
mkdir -p "$DEST/armor-system"
tar -C "$REPO" \
    --exclude='armor-system/data/raw' \
    --exclude='armor-system/data/processed' \
    --exclude='armor-system/airflow/logs' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    -cf - armor-system | tar -C "$DEST" -xf - --strip-components=0

# Record exactly what was deployed. A Space with no provenance is a mystery
# binary six weeks later.
{
  echo "built_from_commit: $(git -C "$REPO" rev-parse HEAD)"
  echo "built_from_branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "built_at_utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "dirty_worktree:    $([ -n "$(git -C "$REPO" status --porcelain)" ] && echo yes || echo no)"
} > "$DEST/COMMIT.txt"

cat > "$DEST/.gitattributes" <<'ATTR'
*.pt filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
ATTR

echo
echo "assembled $(du -sh "$DEST" | cut -f1) in $DEST"
cat "$DEST/COMMIT.txt"
echo
echo "checkpoints deliberately NOT included; they come from MODEL1_HF_REPO at run time"
if find "$DEST" -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' | grep -q .; then
  echo "WARNING: model weights found in the Space directory. Remove them before pushing."
  exit 1
fi
