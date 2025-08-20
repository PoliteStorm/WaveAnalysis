#!/usr/bin/env bash
set -euo pipefail
# Fetch large raw datasets from GitHub Releases into data/ (kept out of git)
# Requires: gh CLI or curl. Verifies sha256sums if provided.
TAG="v1-data"  # change to your release tag
OUTDIR="data/zenodo_5790768"
mkdir -p "$OUTDIR"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }

if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --dir "$OUTDIR"
else
  echo "gh not found; please download files for $TAG from GitHub Releases to $OUTDIR"
fi

# Optional integrity check if sha256sums.txt present next to this script
SUMS="$(dirname "$0")/../data/_manifests/sha256sums.txt"
if [[ -f "$SUMS" ]]; then
  (cd /home/kronos/mushroooom && sha256sum -c "$SUMS") || {
    echo "Checksum mismatch; files may be corrupted."; exit 2; }
fi

echo "Done. Raw files are in $OUTDIR (ignored by git)."
