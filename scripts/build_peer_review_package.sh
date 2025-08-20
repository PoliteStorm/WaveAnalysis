#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y-%m-%dT%H-%M-%S)"
PKG_DIR="$ROOT/results/peer_review/$TS"
OUT_ZIP="$ROOT/results/peer_review/WaveAnalysis_$TS.zip"
mkdir -p "$PKG_DIR"

# 1) Docs
mkdir -p "$PKG_DIR/docs"
cp -f "$ROOT"/docs/{README.md,01_Workflow.md,02_WaveTransform.md,03_Findings_And_Improvements.md,06_Validation_and_Settings.md,08_Results_Interpretation.md} "$PKG_DIR/docs/" 2>/dev/null || true
# Optional extra docs if present
for F in "$ROOT"/docs/*.md; do
  bn="$(basename "$F")"
  [[ -f "$PKG_DIR/docs/$bn" ]] || cp -f "$F" "$PKG_DIR/docs/" || true
done

# 2) Composites and indexes
mkdir -p "$PKG_DIR/composites"
cp -f "$ROOT"/results/zenodo/_composites/species_gallery.png "$PKG_DIR/composites/" 2>/dev/null || true
cp -f "$ROOT"/results/zenodo/_composites/{README.md,csv_index.csv,audits_index.json} "$PKG_DIR/composites/" 2>/dev/null || true

# 3) Latest per-species panels and reports
mkdir -p "$PKG_DIR/species"
for sp in "$ROOT"/results/zenodo/*; do
  [ -d "$sp" ] || continue
  bn="$(basename "$sp")"
  [[ "$bn" == "_composites" ]] && continue
  latest="$(ls -1d "$sp"/* 2>/dev/null | sort | tail -n1)" || true
  [ -n "$latest" ] || continue
  mkdir -p "$PKG_DIR/species/$bn"
  cp -f "$latest"/summary_panel.png "$PKG_DIR/species/$bn/" 2>/dev/null || true
  cp -f "$latest"/report.md "$PKG_DIR/species/$bn/" 2>/dev/null || true
  cp -f "$latest"/audit.md "$PKG_DIR/species/$bn/" 2>/dev/null || true
  cp -f "$latest"/tau_band_timeseries.csv "$PKG_DIR/species/$bn/" 2>/dev/null || true
  cp -f "$latest"/spike_times_s.csv "$PKG_DIR/species/$bn/" 2>/dev/null || true
 done

# 4) Latest ML run (figs + results)
ML_DIR="$ROOT/results/ml"
latest_ml="$(ls -1d "$ML_DIR"/* 2>/dev/null | sort | tail -n1)" || true
if [ -n "${latest_ml:-}" ]; then
  mkdir -p "$PKG_DIR/ml"
  cp -rf "$latest_ml"/figs "$PKG_DIR/ml/" 2>/dev/null || true
  cp -f "$latest_ml"/results.json "$PKG_DIR/ml/" 2>/dev/null || true
  cp -f "$latest_ml"/references.md "$PKG_DIR/ml/" 2>/dev/null || true
fi

# 5) Manifest
cat > "$PKG_DIR/manifest.json" <<JSON
{
  "created_by": "joe knowles",
  "timestamp": "$TS",
  "intended_for": "peer_review",
  "git_sha": "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "paths": {
    "docs": "docs/",
    "composites": "composites/",
    "species": "species/",
    "ml": "ml/"
  }
}
JSON

# 6) Optional PDF (if pandoc exists)
if command -v pandoc >/dev/null 2>&1; then
  (cd "$PKG_DIR/docs" && pandoc -s -o "$PKG_DIR/WaveAnalysis_${TS}.pdf" *.md) || true
fi

# 7) Zip
(cd "$PKG_DIR/.." && zip -qr "$(basename "$OUT_ZIP")" "$(basename "$PKG_DIR")")

echo "$OUT_ZIP"
