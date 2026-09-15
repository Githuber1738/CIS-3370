#!/usr/bin/env bash
# Downloads CIC-DoS2017 into data/cic_dos2017/.
#
# IMPORTANT — read before running:
#   The official UNB CIC page (https://www.unb.ca/cic/datasets/dos-dataset.html)
#   currently shows "We apologize, this dataset is not available at this time."
#   There is no official direct-download URL right now. Two options:
#
#   1. Kaggle mirror (github user dhoogla's repost):
#      https://www.kaggle.com/datasets/dhoogla/cicdos2017
#      Requires a free Kaggle account + API token (~/.kaggle/kaggle.json).
#      This script uses that route via the `kaggle` CLI (pip install kaggle).
#
#   2. Email the Canadian Institute for Cybersecurity directly and ask for
#      access: https://www.unb.ca/cic/about/contact.html
#
#   Cite the original dataset (Jazi et al., "Detecting HTTP-based application
#   layer DoS attacks on web servers in the presence of sampling," Computer
#   Networks, 2017) regardless of which route you use.
set -euo pipefail
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/cic_dos2017"
mkdir -p "$DEST"
cd "$DEST"

if ! command -v kaggle >/dev/null; then
  echo "The 'kaggle' CLI isn't installed. Run: pip install kaggle"
  echo "Then place your Kaggle API token at ~/.kaggle/kaggle.json (from"
  echo "kaggle.com/settings -> API -> Create New Token) and re-run this script."
  exit 1
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "No Kaggle API token found at ~/.kaggle/kaggle.json."
  echo "Get one from kaggle.com/settings -> API -> Create New Token, save it there,"
  echo "then re-run this script. (Falling back to the CIC contact page otherwise:"
  echo "https://www.unb.ca/cic/about/contact.html)"
  exit 1
fi

kaggle datasets download -d dhoogla/cicdos2017 -p "$DEST" --unzip
echo "Done. Files in $DEST:"
ls -lh "$DEST"
