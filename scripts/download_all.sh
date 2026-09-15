#!/usr/bin/env bash
# Runs all three dataset downloads. NSL-KDD is small and safe to run
# unattended; Kitsune and CIC-DoS2017 need a decision/credential first,
# so this only calls download_nsl_kdd.sh automatically.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== NSL-KDD =="
bash "$DIR/download_nsl_kdd.sh"

echo
echo "== Kitsune =="
echo "Not run automatically (large, scenario choice needed)."
echo "Run: bash $DIR/download_kitsune.sh <scenario>   (see script header)"

echo
echo "== CIC-DoS2017 =="
echo "Not run automatically (official host is down, needs a Kaggle API key)."
echo "Run: bash $DIR/download_cic_dos2017.sh   (see script header)"
