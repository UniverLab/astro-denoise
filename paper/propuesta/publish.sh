#!/usr/bin/env bash
# Publish this thesis/proposal PDF to Cloudflare R2 so the UniverLab landing
# always links the latest version. It uploads to a STABLE key, so the public
# URL never changes — just re-run this after editing the document.
#
#   Public URL: https://papers.univerlab.org/astro-denoise/proposal.pdf
#
# It does NOT build the document — build the PDF yourself first, then run this
# to publish it.
#
# Requires: wrangler (via npx) and a one-time `npx wrangler login`.
#
# One-time Cloudflare setup (do once, not on every publish):
#   npx wrangler r2 bucket create univerlab-papers
#   then in the dashboard: R2 → univerlab-papers → Settings → Public access →
#   Connect Custom Domain → papers.univerlab.org
#
# Usage:
#   ./publish.sh                   # uploads the newest PDF in this folder
#   PDF=path/to.pdf ./publish.sh   # override the PDF to upload
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# texforge writes <title>.pdf into this folder — default to the newest PDF here.
PDF="${PDF:-$(ls -t "$HERE"/*.pdf 2>/dev/null | head -1 || true)}"
BUCKET="univerlab-papers"
KEY="astro-denoise/proposal.pdf"

[ -f "$PDF" ] || { echo "error: $PDF not found — build the document first." >&2; exit 1; }

echo "→ uploading $(du -h "$PDF" | cut -f1) to r2://$BUCKET/$KEY"
npx wrangler r2 object put "$BUCKET/$KEY" --file="$PDF" --content-type application/pdf --remote

echo "✓ published → https://papers.univerlab.org/$KEY"
