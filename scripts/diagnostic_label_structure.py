"""Diagnostic Check A (partial) -- label-window structure, no IP data needed.

Confirmed separately: neither OS_Scan_dataset.csv (115 processed statistical
features) nor OS_Scan_labels.csv (binary flag only) contain any IP, MAC, or
packet-flag columns -- that information exists only in the undownloaded
333MB OS_Scan_pcap.pcapng raw capture. This script answers the one part of
Diagnostic A we CAN answer without that: is the label structure a single
contiguous attack-active window (time-range labeling) or scattered
individual flags (per-packet forensic labeling)?
"""

import numpy as np

from src.ingestion.kitsune import load_kitsune


def main():
    df = load_kitsune()
    y = df["label"].to_numpy()

    attack_idx = np.where(y == 1)[0]
    print(f"total rows: {len(y)}")
    print(f"attack rows: {len(attack_idx)}")
    print(f"first attack row: {attack_idx[0]} | last attack row: {attack_idx.max()}")

    # Count contiguous runs of label==1.
    diffs = np.diff(attack_idx)
    breaks = np.where(diffs > 1)[0]
    n_runs = len(breaks) + 1
    print(f"number of contiguous attack-label runs: {n_runs}")
    if n_runs <= 5:
        run_starts = [attack_idx[0]] + [attack_idx[b + 1] for b in breaks]
        run_ends = [attack_idx[b] for b in breaks] + [attack_idx[-1]]
        for s, e in zip(run_starts, run_ends):
            print(f"  run: rows {s}-{e} ({e - s + 1} rows)")
    else:
        print(f"  (showing first 10 run boundaries out of {n_runs})")
        run_starts = [attack_idx[0]] + [attack_idx[b + 1] for b in breaks[:9]]
        run_ends = [attack_idx[b] for b in breaks[:9]]
        for s, e in zip(run_starts, run_ends):
            print(f"  run: rows {s}-{e} ({e - s + 1} rows)")

    span = attack_idx.max() - attack_idx[0] + 1
    density = len(attack_idx) / span
    print(f"\nattack window span: {span} rows | density of label==1 within that span: {density:.4f}")


if __name__ == "__main__":
    main()
