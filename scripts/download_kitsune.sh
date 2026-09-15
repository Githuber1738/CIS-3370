#!/usr/bin/env bash
# Downloads the Kitsune Network Attack Dataset into data/kitsune/.
#
# IMPORTANT — read before running:
#   The full dataset is 17.7 GB (UCI ML Repository, CC BY 4.0, no login
#   required): https://archive.ics.uci.edu/static/public/516/kitsune+network+attack+dataset.zip
#   That is almost certainly more than you need or want to download blind.
#
#   The dataset is really 9 separate attack scenarios, each with its own
#   feature CSV + label file, ranging ~1.9-3 GB apiece (SSDP Flood, ARP MitM,
#   Video Injection, SYN DoS, Active Wiretap, and others). For the xNIDS
#   capstone reproduction you almost certainly only need ONE scenario to train
#   Kitsune's autoencoder ensemble (KitNET) and reproduce Table 12's
#   Kitsune row (Precision 0.9568 / Recall 0.9972 / FPR 0.001) — check which
#   scenario the xNIDS authors' released code (github.com/CactiLab/code-xNIDS)
#   actually trains on before you burn bandwidth/disk on all nine.
#
# Usage:
#   ./download_kitsune.sh full        # everything, 17.7 GB
#   ./download_kitsune.sh <scenario>  # e.g. ./download_kitsune.sh syn_dos
set -euo pipefail
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/kitsune"
mkdir -p "$DEST"
cd "$DEST"

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 [full|<scenario-name>]"
  echo "Nothing downloaded. See the comments at the top of this script."
  exit 1
fi

if [[ "$MODE" == "full" ]]; then
  echo "Downloading the full 17.7 GB archive. This will take a while."
  curl -fSL "https://archive.ics.uci.edu/static/public/516/kitsune+network+attack+dataset.zip" \
    -o kitsune_full.zip
  unzip -o kitsune_full.zip
else
  echo "Per-scenario direct links aren't stable enough to hardcode reliably."
  echo "Go to https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset"
  echo "or https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune,"
  echo "pick the '$MODE' scenario's feature CSV + labels file, and drop them in:"
  echo "  $DEST/$MODE/"
  mkdir -p "$DEST/$MODE"
fi
