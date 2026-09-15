#!/usr/bin/env bash
# Downloads NSL-KDD into data/nsl_kdd/.
#
# The official UNB CIC page (https://www.unb.ca/cic/datasets/nsl.html) currently
# shows "We apologize, this dataset is no longer available." As of this setup,
# no official direct-download URL exists. This script pulls from the
# defcom17/NSL_KDD GitHub mirror instead, which carries the same files
# (KDDTrain+.txt, KDDTest+.txt, the 20% training subset, KDDTest-21, and the
# original ARFF zip). Cite the original dataset (Tavallaee et al., NSL-KDD,
# 2009) in your report regardless of where you pulled the bytes from, and note
# in your Wiki report that the official host was down and you used a mirror.
set -euo pipefail
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/nsl_kdd"
BASE="https://raw.githubusercontent.com/defcom17/NSL_KDD/master"

mkdir -p "$DEST"
cd "$DEST"

files=(
  "KDDTrain%2B.txt:KDDTrain+.txt"
  "KDDTest%2B.txt:KDDTest+.txt"
  "KDDTrain%2B_20Percent.txt:KDDTrain+_20Percent.txt"
  "KDDTest-21.txt:KDDTest-21.txt"
  "Original%20NSL%20KDD%20Zip.zip:Original_NSL_KDD.zip"
)

for entry in "${files[@]}"; do
  remote="${entry%%:*}"
  local="${entry##*:}"
  echo "Fetching $local ..."
  curl -fSL "$BASE/$remote" -o "$local"
done

if command -v unzip >/dev/null; then
  unzip -o Original_NSL_KDD.zip -d original_arff >/dev/null || true
fi

echo "Done. Files in $DEST:"
ls -lh "$DEST"
