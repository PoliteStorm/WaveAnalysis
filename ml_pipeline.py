#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import datetime as _dt

# Reuse functions from prove_transform.py
import prove_transform as pt


def _compute_k_stats(k: np.ndarray, P: np.ndarray):
    total = float(np.sum(P) + 1e-12)
    centroid = float(np.sum(k * P) / total)
    var = float(np.sum(((k - centroid) ** 2) * P) / total)
    bw = float(np.sqrt(max(var, 0.0)))
    # Peak count: local maxima above percentile threshold
    if P.size >= 3:
        thr = np.percentile(P, 90.0)
        peaks = 0
        for i in range(1, len(P) - 1):
            if P[i] > P[i - 1] and P[i] > P[i + 1] and P[i] >= thr:
                peaks += 1
    else:
        peaks = 0
    return centroid, bw, float(peaks)


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad - ((w + 1) % 2)), mode='edge')
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(xp, kernel, mode='valid')
    if y.shape[0] != x.shape[0]:
        y = y[:x.shape[0]]
    return y


def _stats_entropy(vals: np.ndarray, nbins: int = 20):
    out = [0.0, 0.0, 0.0, 0.0]  # entropy, skew, kurt_excess, mean
    n = vals.size
    if n == 0:
        return out
    x = vals.astype(float)
    # entropy
    hist, _ = np.histogram(x, bins=nbins, density=False)
    p = hist.astype(float) / (np.sum(hist) + 1e-12)
    nz = p[p > 0]
    H = -np.sum(nz * np.log2(nz))
    mu = float(np.mean(x))
    sig = float(np.std(x))
    if sig > 0:
        z = (x - mu) / sig
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4)) - 3.0
    else:
        skew, kurt = 0.0, 0.0
    return [float(H), skew, kurt, mu]


def _compute_spike_features(V: np.ndarray, fs_hz: float = 1.0, min_amp_mV: float = 0.1, min_isi_s: float = 120.0, baseline_win_s: float = 600.0):
    # Simple absolute threshold on detrended signal
    w = max(1, int(round(baseline_win_s * fs_hz)))
    base = _moving_average(V, w)
    x = V - base
    thr = float(min_amp_mV)
    idx = np.where(np.abs(x) >= thr)[0]
    spikes_t = []
    spikes_a = []
    if idx.size > 0:
        groups = []
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
            else:
                groups.append((start, prev))
                start = k
                prev = k
        groups.append((start, prev))
        # refractory
        min_gap = int(round(min_isi_s * fs_hz))
        filt = []
        last_end = -10**9
        for a, b in groups:
            if a - last_end < min_gap:
                if filt:
                    filt[-1] = (filt[-1][0], b)
                else:
                    filt.append((a, b))
            else:
                filt.append((a, b))
            last_end = filt[-1][1]
        for a, b in filt:
            seg = x[a:b+1]
            if seg.size == 0:
                continue
            pk_local = int(np.argmax(np.abs(seg)))
            pk = a + pk_local
            spikes_t.append(pk / fs_hz)
            spikes_a.append(float(V[pk]))
    spikes_t = np.array(spikes_t, dtype=float)
    spikes_a = np.array(spikes_a, dtype=float)
    isi = np.diff(spikes_t) if spikes_t.size >= 2 else np.array([], dtype=float)
    # Rate per hour
    duration_h = float(len(V) / fs_hz / 3600.0) if len(V) > 0 else 0.0
    rate_per_h = float(spikes_t.size / duration_h) if duration_h > 0 else 0.0
    # Stats
    H_amp, skew_amp, kurt_amp, mean_amp = _stats_entropy(spikes_a)
    H_isi, skew_isi, kurt_isi, mean_isi = _stats_entropy(isi)
    return [rate_per_h, H_amp, skew_amp, kurt_amp, mean_amp, H_isi, skew_isi, kurt_isi, mean_isi]


def extract_features_from_file(path: str, tau_values, nu0: int, channel_hint: str = None, n_u: int = 1024):
    t, channels = pt.load_zenodo_timeseries(path)
    # Pick channel
    pick = None
    if channel_hint and channel_hint in channels:
        pick = channel_hint
    else:
        for name, vec in channels.items():
            if np.isfinite(vec).any():
                pick = name
                break
    if pick is None:
        raise RuntimeError(f"No finite channel in {path}")
    V = channels[pick]

    def V_func(t_vals):
        return np.interp(t_vals, t, np.nan_to_num(V, nan=np.nanmean(V)))

    U_max = np.sqrt(t[-1]) if len(t) > 1 else 1.0
    u0_grid = np.linspace(0.0, U_max, nu0, endpoint=False)
    # Shared u-grid for transforms
    N_u = int(n_u)
    u_grid = np.linspace(0.0, U_max, N_u, endpoint=False)

    def _k_entropy(kv: np.ndarray, Pv: np.ndarray, nbins: int = 32) -> float:
        # Shannon entropy (bits) of coarse k-distribution
        if Pv.size < 2:
            return 0.0
        edges = np.linspace(kv.min(), kv.max(), nbins + 1)
        idx = np.clip(np.digitize(kv, edges) - 1, 0, nbins - 1)
        bins = np.bincount(idx, weights=Pv, minlength=nbins).astype(float)
        p = bins / (np.sum(bins) + 1e-12)
        nz = p[p > 0]
        H = -np.sum(nz * np.log2(nz))
        return float(H)

    # Features per window u0: for each tau: [norm_power, k_centroid, k_bandwidth, peak_count, k_entropy]
    feats_list = []
    for u0 in u0_grid:
        window_feats = []
        power_per_tau = []
        k_stats_tmp = []
        for tau in tau_values:
            k_fft, W = pt.sqrt_time_transform_fft(V_func, tau, u_grid, u0=u0)
            P = np.abs(W) ** 2
            power = float(np.sum(P))
            c, bw, peaks = _compute_k_stats(k_fft, P)
            Hk = _k_entropy(k_fft, P)
            power_per_tau.append(power)
            k_stats_tmp.append((c, bw, peaks, Hk))
        # Normalize powers across tau for this window
        power_arr = np.array(power_per_tau, dtype=float)
        norm = float(np.sum(power_arr) + 1e-12)
        power_norm = (power_arr / norm).tolist()
        # Append in tau order
        for i in range(len(tau_values)):
            c, bw, peaks, Hk = k_stats_tmp[i]
            window_feats += [power_norm[i], c, bw, peaks, Hk]
        # Add cheap power ratios (assuming three taus: fast, slow, very slow)
        if len(power_norm) >= 3:
            f, s, vs = power_norm[0], power_norm[1], power_norm[2]
            eps = 1e-9
            window_feats += [
                float(f / (s + eps)),
                float(s / (vs + eps)),
                float(f / (vs + eps)),
            ]
        feats_list.append(window_feats)
    feats = np.array(feats_list, dtype=float)
    # File-level aggregate (mean/std per feature) and append to each row
    mean_feats = np.mean(feats, axis=0)
    std_feats = np.std(feats, axis=0)
    feats = np.hstack([feats, np.tile(mean_feats, (feats.shape[0], 1)), np.tile(std_feats, (feats.shape[0], 1))])
    # Append spike-derived features (same per row; cheap to compute once per file)
    spike_feats = _compute_spike_features(np.nan_to_num(V, nan=np.nanmean(V)), fs_hz=1.0, min_amp_mV=0.1, min_isi_s=120.0, baseline_win_s=600.0)
    spike_block = np.tile(np.array(spike_feats, dtype=float), (feats.shape[0], 1))
    feats = np.hstack([feats, spike_block])
    times = u0_grid ** 2
    return feats, times, pick


def simple_train_test_split(X, y, test_frac=0.3, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(test_frac * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_linear_classifier(X_train, y_train):
    # One-vs-rest linear classifier via ridge regression (closed form)
    classes = np.unique(y_train)
    C = len(classes)
    n, d = X_train.shape
    Y = np.zeros((n, C), dtype=float)
    for i, c in enumerate(classes):
        Y[:, i] = (y_train == c).astype(float)
    # Add bias
    Xb = np.hstack([X_train, np.ones((n, 1))])
    # Ridge parameter
    lam = 1e-3
    A = Xb.T @ Xb + lam * np.eye(d + 1)
    W = np.linalg.solve(A, Xb.T @ Y)
    return classes, W


def predict_linear_classifier(X, classes, W):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    scores = Xb @ W
    idx = np.argmax(scores, axis=1)
    return classes[idx]


def train_random_forest_if_available(X_train, y_train):
    try:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
        return clf
    except Exception:
        return None


def predict_with_model(model, X):
    try:
        return model.predict(X)
    except Exception:
        return None


def train_logistic_if_available(X_train, y_train):
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=200, multi_class='ovr', random_state=0)
        clf.fit(X_train, y_train)
        return clf
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="ML pipeline on √t features for fungal species discrimination")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--nu0", type=int, default=128)
    parser.add_argument("--taus", type=str, default="5.5,24.5,104")
    parser.add_argument("--channel", type=str, default="")
    parser.add_argument("--json_out", type=str, default="/home/kronos/mushroooom/ml_results.json")
    parser.add_argument("--n_u", type=int, default=1024, help="Number of u samples for FFT per window")
    parser.add_argument("--cache_dir", type=str, default="/home/kronos/mushroooom/cache/features")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--lofo", action="store_true", help="Leave-one-file-out CV")
    parser.add_argument("--loco", action="store_true", help="Leave-one-channel-out CV")
    parser.add_argument("--out_dir", type=str, default="", help="If set and --json_out empty, write results to timestamped subfolder here")
    parser.add_argument("--progress", action="store_true", help="Print progress status messages")
    parser.add_argument("--only_file", type=str, default="", help="Process only this filename (exact match within data_dir)")
    parser.add_argument("--plot_ml", action="store_true", help="Save ML diagnostics plots (feature importance, confusion matrix, calibration)")
    args = parser.parse_args()

    tau_values = [float(x) for x in args.taus.split(",") if x.strip()]
    def log(msg: str):
        if args.progress:
            print(f"[STATUS] {msg}", flush=True)

    # Map files to species labels by filename
    species_map = {}
    X_list = []
    y_list = []
    groups_file = []     # file index per row for LOFO
    groups_channel = []  # channel index per row for LOCO
    file_summaries = []
    os.makedirs(args.cache_dir, exist_ok=True)

    log("Listing data files…")
    data_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith('.txt')]
    if args.only_file:
        data_files = [f for f in data_files if f == args.only_file]
    data_files.sort()
    sample_idx = 0
    for file_idx, fname in enumerate(data_files):
        path = os.path.join(args.data_dir, fname)
        base = os.path.splitext(fname)[0]
        log(f"File {file_idx+1}/{len(data_files)}: {fname}")
        if 'Schizophyllum' in base:
            label = 'Schizophyllum_commune'
        elif 'Flammulina' in base or 'Enoki' in base:
            label = 'Flammulina_velutipes'
        elif 'Omphalotus' in base or 'Ghost' in base:
            label = 'Omphalotus_nidiformis'
        elif 'Cordyceps' in base:
            label = 'Cordyceps_militaris'
        else:
            label = base.replace(' ', '_')
        # Load channels and iterate all finite ones unless --channel specified
        t_tmp, chans_dict = pt.load_zenodo_timeseries(path)
        chan_names = [args.channel] if args.channel and args.channel in chans_dict else list(chans_dict.keys())
        for chan in chan_names:
            vec = chans_dict.get(chan)
            if vec is None or not np.isfinite(vec).any():
                continue
            log(f"  Channel: {chan}")
            cache_key = f"{base.replace(' ','_')}_{chan}_nu0{args.nu0}_nu{args.n_u}_taus{('-').join([str(t) for t in tau_values])}.npz"
            cache_path = os.path.join(args.cache_dir, cache_key)
            if (not args.force_recompute) and os.path.exists(cache_path):
                log("    Cache hit")
                data = np.load(cache_path)
                feats = data['feats']
                times = data['times']
                channel_used = str(data['channel'])
            else:
                log("    Extracting features…")
                feats, times, channel_used = extract_features_from_file(path, tau_values, args.nu0, channel_hint=chan, n_u=args.n_u)
                # cast to float32 to reduce memory footprint in cache
                feats32 = feats.astype(np.float32, copy=False)
                np.savez_compressed(cache_path, feats=feats32, times=times, channel=channel_used)
            log(f"    Segments: {feats.shape[0]}, Features: {feats.shape[1]}")

            X_list.append(feats)
            y_list.append(np.full((feats.shape[0],), label, dtype='<U64'))
            groups_file.append(np.full((feats.shape[0],), file_idx, dtype=int))
            groups_channel.append(np.full((feats.shape[0],), sample_idx, dtype=int))
            file_summaries.append({
                'file': path,
                'label': label,
                'channel': channel_used,
                'segments': int(feats.shape[0]),
            })
            sample_idx += 1

    if not X_list:
        raise SystemExit("No .txt files found in data_dir")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups_file = np.concatenate(groups_file)
    groups_channel = np.concatenate(groups_channel)

    # Training/evaluation
    results = {}
    y_true_all = []
    y_pred_all = []
    proba_all = []
    brier_sum = 0.0
    brier_n = 0
    if args.lofo or args.loco:
        # Leave-one-group-out CV (file or channel)
        if args.lofo:
            group_vec = groups_file
            results['cv'] = 'leave_one_file_out'
        else:
            group_vec = groups_channel
            results['cv'] = 'leave_one_channel_out'
        uniq_groups = np.unique(group_vec)
        fold_acc = []
        for gid in uniq_groups:
            log(f"Training fold (leave group {int(gid)} out)…")
            test_mask = (group_vec == gid)
            train_mask = ~test_mask
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            # Try logistic first for small-sample generalization
            clf = train_logistic_if_available(X_train, y_train)
            if clf is None:
                clf = train_random_forest_if_available(X_train, y_train)
                model_name = 'RandomForestClassifier'
            else:
                model_name = 'LogisticRegression'
            y_pred = predict_with_model(clf, X_test)
            # probabilities if available
            try:
                proba = clf.predict_proba(X_test)
            except Exception:
                proba = None
            acc_f = float(np.mean(y_pred == y_test))
            fold_acc.append(acc_f)
            log(f"  Fold accuracy: {acc_f:.3f}")
            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            if proba is not None:
                proba_all.append(proba)
                # per-fold Brier score (one-vs-rest multi-class)
                try:
                    import numpy as _np
                    classes_fold = getattr(clf, 'classes_', _np.unique(y_train))
                    class_to_idx = {str(c): i for i, c in enumerate(classes_fold)}
                    y_idx = _np.array([class_to_idx.get(str(c), -1) for c in y_test])
                    # build one-hot
                    Y = _np.zeros_like(proba)
                    for i in range(len(y_idx)):
                        j = y_idx[i]
                        if 0 <= j < Y.shape[1]:
                            Y[i, j] = 1.0
                    brier_fold = float(_np.mean(_np.sum((proba - Y) ** 2, axis=1)))
                    brier_sum += brier_fold * len(y_test)
                    brier_n += len(y_test)
                except Exception:
                    pass
        acc = float(np.mean(fold_acc)) if fold_acc else 0.0
        results['fold_accuracies'] = [float(a) for a in fold_acc]
    else:
        # Simple split
        log("Training random 70/30 split…")
        X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_frac=0.3)
        clf = train_logistic_if_available(X_train, y_train)
        if clf is None:
            clf = train_random_forest_if_available(X_train, y_train)
            model_name = 'RandomForestClassifier'
        else:
            model_name = 'LogisticRegression'
        y_pred = predict_with_model(clf, X_test)
        try:
            proba = clf.predict_proba(X_test)
        except Exception:
            proba = None
        acc = float(np.mean(y_pred == y_test))
        results['cv'] = 'random_split_70_30'
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        if proba is not None:
            proba_all.append(proba)
            try:
                import numpy as _np
                classes_fold = getattr(clf, 'classes_', _np.unique(y_train))
                class_to_idx = {str(c): i for i, c in enumerate(classes_fold)}
                y_idx = _np.array([class_to_idx.get(str(c), -1) for c in y_test])
                Y = _np.zeros_like(proba)
                for i in range(len(y_idx)):
                    j = y_idx[i]
                    if 0 <= j < Y.shape[1]:
                        Y[i, j] = 1.0
                brier_fold = float(_np.mean(_np.sum((proba - Y) ** 2, axis=1)))
                brier_sum += brier_fold * len(y_test)
                brier_n += len(y_test)
            except Exception:
                pass

    timestamp = _dt.datetime.now().isoformat(timespec='seconds')
    result = {
        'tau_values': tau_values,
        'nu0': int(args.nu0),
        'n_u': int(args.n_u),
        'data_dir': args.data_dir,
        'files': file_summaries,
        'num_samples': int(sample_idx),
        'accuracy': acc,
        'model': model_name,
        'classes': [str(c) for c in np.unique(y)],
        'created_by': 'joe knowles',
        'timestamp': timestamp,
        'intended_for': 'peer_review',
        **results,
    }
    if brier_n > 0:
        result['brier_score'] = float(brier_sum / brier_n)
    # Determine target output paths
    target_json = args.json_out
    if (not target_json) and args.out_dir:
        ts = timestamp.replace(":", "-")
        cv_tag = results.get('cv', 'split')
        slug = f"{cv_tag}_nu0{args.nu0}_nu{args.n_u}"
        out_dir = os.path.join(args.out_dir, ts + "_" + slug)
        os.makedirs(out_dir, exist_ok=True)
        target_json = os.path.join(out_dir, 'results.json')
    elif target_json:
        out_dir = os.path.dirname(target_json)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "/home/kronos/mushroooom/results/ml"
        os.makedirs(out_dir, exist_ok=True)
        target_json = os.path.join(out_dir, 'results.json')

    log("Writing results…")
    with open(target_json, 'w') as f:
        json.dump(result, f)
    # Bibliography alongside
    bib = os.path.join(out_dir, 'references.md')
    with open(bib, 'w') as f:
        f.write("- On spiking behaviour of Pleurotus djamor (Sci Rep 2018): https://www.nature.com/articles/s41598-018-26007-1?utm_source=chatgpt.com\n")
        f.write("- Multiscalar electrical spiking in Schizophyllum commune (Sci Rep 2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/?utm_source=chatgpt.com#Sec2\n")
        f.write("- Language of fungi derived from electrical spiking activity (R. Soc. Open Sci. 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/?utm_source=chatgpt.com\n")
        f.write("- Electrical response of fungi to changing moisture content (Fungal Biol Biotech 2023): https://fungalbiolbiotech.biomedcentral.com/articles/10.1186/s40694-023-00155-0?utm_source=chatgpt.com\n")
        f.write("- Electrical activity of fungi: Spikes detection and complexity analysis (Biosystems 2021): https://www.sciencedirect.com/science/article/pii/S0303264721000307\n")
    print(json.dumps({'json': target_json, 'bib': bib}))

    # Optional ML diagnostics plots
    if args.plot_ml:
        # Prepare output dir for figures
        figs_dir = os.path.join(out_dir, 'figs')
        os.makedirs(figs_dir, exist_ok=True)
        # Feature importance / coefficients
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            plt = None
        if plt is not None:
            try:
                if hasattr(clf, 'feature_importances_'):
                    imp = np.asarray(clf.feature_importances_, dtype=float)
                elif hasattr(clf, 'coef_'):
                    coef = np.asarray(clf.coef_, dtype=float)
                    imp = np.mean(np.abs(coef), axis=0)
                else:
                    imp = None
                if imp is not None:
                    idx = np.argsort(imp)[::-1][:50]
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
                    ax.barh(range(len(idx)), imp[idx][::-1], color='slateblue')
                    ax.set_yticks(range(len(idx)))
                    ax.set_yticklabels([f'f{j}' for j in idx[::-1]], fontsize=7)
                    ax.set_xlabel('importance')
                    ax.set_title('Feature importance (top 50)')
                    fig.tight_layout()
                    fig.savefig(os.path.join(figs_dir, 'feature_importance.png'))
                    plt.close(fig)
            except Exception:
                pass
            # Confusion matrix
            try:
                from sklearn.metrics import confusion_matrix
                y_true_c = np.concatenate(y_true_all) if y_true_all else np.array([])
                y_pred_c = np.concatenate(y_pred_all) if y_pred_all else np.array([])
                if y_true_c.size and y_pred_c.size:
                    labels = np.unique(np.concatenate([y_true_c, y_pred_c]))
                    cm = confusion_matrix(y_true_c, y_pred_c, labels=labels)
                    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=140)
                    im = ax.imshow(cm, cmap='Blues')
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                    ax.set_yticks(range(len(labels)))
                    ax.set_yticklabels(labels, fontsize=7)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7)
                    ax.set_title('Confusion matrix')
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    fig.savefig(os.path.join(figs_dir, 'confusion_matrix.png'))
                    plt.close(fig)
            except Exception:
                pass
            # Calibration curve (max prob vs correctness)
            try:
                if proba_all:
                    P = np.vstack(proba_all)
                    y_true_c = np.concatenate(y_true_all)
                    # correctness of predicted class
                    y_pred_idx = np.argmax(P, axis=1)
                    classes_list = result['classes']
                    # map y_true to indices
                    class_to_idx = {c: i for i, c in enumerate(classes_list)}
                    y_true_idx = np.array([class_to_idx.get(str(c), -1) for c in y_true_c])
                    correct = (y_pred_idx == y_true_idx)
                    maxp = np.max(P, axis=1)
                    # bin into 10 bins
                    bins = np.linspace(0.0, 1.0, 11)
                    inds = np.digitize(maxp, bins) - 1
                    xs = []
                    ys = []
                    for b in range(10):
                        mask = inds == b
                        if np.any(mask):
                            xs.append(np.mean(maxp[mask]))
                            ys.append(np.mean(correct[mask].astype(float)))
                    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=140)
                    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
                    ax.plot(xs, ys, marker='o', color='darkorange', label='model')
                    ax.set_xlabel('predicted probability (max class)')
                    ax.set_ylabel('empirical accuracy')
                    ax.set_title('Calibration')
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.join(figs_dir, 'calibration.png'))
                    plt.close(fig)
            except Exception:
                pass


if __name__ == '__main__':
    main()


