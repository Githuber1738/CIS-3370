"""IMPROVEMENT ATTEMPT #4 -- the real KitNET architecture (ensemble of small
per-channel autoencoders + an output-layer autoencoder combining them),
in place of xNIDS's substitute single monolithic autoencoder.

Real KitNET (Mirsky et al.) normally discovers feature clusters online via
incremental correlation tracking, since a real deployment doesn't know
feature semantics ahead of time. We already know the true semantic
grouping -- confirmed against xNIDS's own kitsune_group_sizes/names in
Scripts/explanation.py -- so we use those 5 channels directly (MAC-IP,
IP, Jitter, IPtoIP, Socket) rather than re-deriving them. That keeps the
architectural idea (many small autoencoders, each seeing only a related
feature cluster, combined by an output autoencoder) faithful while
skipping a clustering step whose answer we already have.

Same chronological split and scaling as the faithful reproduction, so
this isolates exactly one variable: ensemble-of-small-models vs. one
monolithic model.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras

from src.ingestion.kitsune import load_kitsune
from src.models.kitsune_autoencoder import MODEL_DIR, evaluate

SPLIT_INDEX = 1_000_000
GROUP_SIZES = [15, 15, 15, 35, 35]
GROUP_NAMES = ["MAC-IP", "IP", "Jitter", "IPtoIP", "Socket"]


def build_small_autoencoder(input_dim: int):
    hidden = max(4, input_dim // 2)
    bottleneck = max(2, input_dim // 4)
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(hidden, activation="relu")(inp)
    x = keras.layers.Dense(bottleneck, activation="relu")(x)
    x = keras.layers.Dense(hidden, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def group_slices():
    slices = []
    start = 0
    for size in GROUP_SIZES:
        slices.append(slice(start, start + size))
        start += size
    return slices


def main():
    print("Loading Kitsune OS Scan data...")
    df = load_kitsune()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test = X[:SPLIT_INDEX], X[SPLIT_INDEX:]
    y_test = y[SPLIT_INDEX:]

    robust = RobustScaler().fit(X)
    X_train_s = robust.transform(X_train)
    X_test_s = robust.transform(X_test)
    minmax = MinMaxScaler().fit(np.vstack([X_train_s, X_test_s]))
    X_train_s = minmax.transform(X_train_s)
    X_test_s = minmax.transform(X_test_s)

    slices = group_slices()
    train_group_rmse = np.zeros((len(X_train_s), len(slices)))
    test_group_rmse = np.zeros((len(X_test_s), len(slices)))

    for i, (sl, name, size) in enumerate(zip(slices, GROUP_NAMES, GROUP_SIZES)):
        print(f"Training small autoencoder for channel '{name}' ({size} features)...")
        ae = build_small_autoencoder(size)
        ae.fit(X_train_s[:, sl], X_train_s[:, sl], epochs=10, batch_size=5000, verbose=0)

        train_recon = ae.predict(X_train_s[:, sl], verbose=0)
        train_group_rmse[:, i] = np.sqrt(np.mean(np.square(X_train_s[:, sl] - train_recon), axis=1))
        test_recon = ae.predict(X_test_s[:, sl], verbose=0)
        test_group_rmse[:, i] = np.sqrt(np.mean(np.square(X_test_s[:, sl] - test_recon), axis=1))

    # Save the per-group RMSE features for the supervised-hybrid follow-up.
    np.save(MODEL_DIR.parent / "artifacts" / "kitsune_train_group_rmse.npy", train_group_rmse)
    np.save(MODEL_DIR.parent / "artifacts" / "kitsune_test_group_rmse.npy", test_group_rmse)

    print("Training output-layer autoencoder on the 5 per-channel RMSEs...")
    rmse_scaler = MinMaxScaler().fit(np.vstack([train_group_rmse, test_group_rmse]))
    train_rmse_s = rmse_scaler.transform(train_group_rmse)
    test_rmse_s = rmse_scaler.transform(test_group_rmse)

    output_ae = build_small_autoencoder(len(slices))
    output_ae.fit(train_rmse_s, train_rmse_s, epochs=20, batch_size=5000, verbose=0)

    train_final_recon = output_ae.predict(train_rmse_s, verbose=0)
    train_final_score = np.mean(np.square(train_rmse_s - train_final_recon), axis=1)
    threshold = np.mean(train_final_score) + 2 * np.std(train_final_score)

    test_final_recon = output_ae.predict(test_rmse_s, verbose=0)
    test_final_score = np.mean(np.square(test_rmse_s - test_final_recon), axis=1)
    y_pred = (test_final_score > threshold).astype(int)

    metrics = evaluate(y_test, y_pred)
    print(f"\nthreshold: {threshold}")
    print("\n--- Results (KitNET-style ensemble, chronological split) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nBaseline (single monolithic autoencoder): Precision 0.500 / Recall 1.000 / FPR 0.104")
    print("xNIDS paper's reported Kitsune row (Table 12): Precision 0.9568 / Recall 0.9972 / FPR 0.001")


if __name__ == "__main__":
    main()
