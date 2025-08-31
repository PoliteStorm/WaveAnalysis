#!/usr/bin/env python3
import os
import glob
import json
import datetime as _dt
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import wavfile

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import prove_transform as pt


def sliding_frames(n: int, fs: int, win_s: float, hop_s: float) -> List[Tuple[int, int]]:
    win = max(1, int(win_s * fs))
    hop = max(1, int(hop_s * fs))
    frames = []
    start = 0
    while start + win <= n:
        frames.append((start, start + win))
        start += hop
    if not frames and n > 0:
        frames.append((0, n))
    return frames


def compute_mfcc_frames(audio: np.ndarray, fs: int, win_s: float, hop_s: float, n_mfcc: int = 20, force_basic: bool = False) -> np.ndarray:
    if not force_basic:
        try:
            import librosa
            y = audio.astype(np.float32)
            mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=n_mfcc, n_fft=int(win_s*fs), hop_length=int(hop_s*fs), htk=True)
            return mfcc.T.astype(np.float32)
        except Exception:
            pass
    # Basic spectral stats per frame
    n = len(audio)
    frames = sliding_frames(n, fs, win_s, hop_s)
    feats = []
    for s, e in frames:
        x = audio[s:e]
        x = x - np.mean(x)
        if len(x) < 4:
            feats.append([0.0]*6)
            continue
        w = 0.5*(1-np.cos(2*np.pi*np.arange(len(x))/max(1,len(x)-1)))
        X = np.fft.rfft(x*w)
        psd = np.abs(X)**2
        freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
        psd_sum = np.sum(psd)+1e-12
        centroid = float(np.sum(freqs*psd)/psd_sum)
        var = np.sum(((freqs-centroid)**2)*psd)/psd_sum
        bw = float(np.sqrt(max(var,0.0)))
        gm = np.exp(np.mean(np.log(psd+1e-12)))
        am = np.mean(psd+1e-12)
        flat = float(gm/am)
        zcr = float(np.mean(np.diff(np.signbit(x))!=0))
        rms = float(np.sqrt(np.mean(x*x)))
        feats.append([rms, zcr, centroid, bw, flat, float(np.max(x)-np.min(x))])
    return np.asarray(feats, dtype=np.float32)


def signal_window_features(V: np.ndarray, fs_signal: float, t0: float, t1: float) -> np.ndarray:
    # Interpolate onto uniform grid and compute simple stats
    if t1 <= t0:
        return np.zeros(6, dtype=np.float32)
    t_axis = np.arange(len(V), dtype=np.float32)/float(fs_signal)
    grid = np.linspace(t0, t1, 1024, dtype=np.float32)
    v = np.interp(grid, t_axis, V).astype(np.float32)
    v = v - np.mean(v)
    if v.size < 4:
        return np.zeros(6, dtype=np.float32)
    # Time stats
    std = float(np.std(v))
    abs_deriv = float(np.mean(np.abs(np.diff(v))))
    rng = float(np.max(v)-np.min(v))
    # Simple spectrum
    w = 0.5*(1-np.cos(2*np.pi*np.arange(len(v))/max(1,len(v)-1)))
    X = np.fft.rfft(v*w)
    psd = np.abs(X)**2
    freqs = np.fft.rfftfreq(len(v), d=1.0/fs_signal)
    psd_sum = np.sum(psd)+1e-12
    centroid = float(np.sum(freqs*psd)/psd_sum)
    var = np.sum(((freqs-centroid)**2)*psd)/psd_sum
    bw = float(np.sqrt(max(var,0.0)))
    flat = float(np.exp(np.mean(np.log(psd+1e-12)))/np.mean(psd+1e-12))
    return np.asarray([std, abs_deriv, rng, centroid, bw, flat], dtype=np.float32)


def build_signal_matrix(V: np.ndarray, fs_signal: float, n_frames: int, win_s: float, hop_s: float, speed: float) -> np.ndarray:
    # Mirror audio framing: frame i spans [i*hop, i*hop+win] in audio seconds → map to signal secs via speed
    feats = []
    for i in range(n_frames):
        t0_a = i*hop_s
        t1_a = t0_a + win_s
        t0_s = t0_a*speed
        t1_s = t1_a*speed
        feats.append(signal_window_features(V, fs_signal, t0_s, t1_s))
    return np.asarray(feats, dtype=np.float32)


def per_run_cca(meta_path: str, win_s: float, hop_s: float, n_mfcc: int, n_components: int, perm_iters: int = 200, force_basic_audio: bool = True) -> Dict:
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    wav_path = meta['paths'].get('wav') or meta['paths'].get('mp3')
    if not wav_path or not os.path.exists(wav_path):
        return {'error': True, 'reason': 'missing_audio', 'meta': meta_path}
    fs_a, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if audio.dtype.kind in ('i','u'):
        audio = audio.astype(np.float32)/max(1, np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)

    # Compute audio MFCC/features
    A = compute_mfcc_frames(audio, fs_a, win_s, hop_s, n_mfcc=n_mfcc, force_basic=force_basic_audio)
    if A is None or not hasattr(A, 'shape'):
        # Retry with forced basic features
        A = compute_mfcc_frames(audio, fs_a, win_s, hop_s, n_mfcc=n_mfcc, force_basic=True)
    if A is None or not hasattr(A, 'shape'):
        return {'error': True, 'reason': 'audio_features_none', 'meta': meta_path}
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    n_frames = int(A.shape[0])
    if n_frames <= 0:
        return {'error': True, 'reason': 'no_frames', 'meta': meta_path}
    if n_frames < 5:
        return {'error': True, 'reason': 'too_few_frames', 'meta': meta_path}

    # Load raw signal and map windows
    t, channels = pt.load_zenodo_timeseries(meta['file'])
    V = np.asarray(channels[meta['channel']], dtype=np.float32)
    fs_sig = float(meta.get('fs_hz', 1.0))
    speed = float(meta.get('speed', 3600.0))
    S = build_signal_matrix(V, fs_sig, n_frames, win_s, hop_s, speed)

    # Align sizes
    m = min(len(A), len(S))
    A = A[:m]
    S = S[:m]

    # Drop any rows with NaNs
    mask = np.isfinite(A).all(axis=1) & np.isfinite(S).all(axis=1)
    A = A[mask]
    S = S[mask]
    if len(A) < 5:
        return {'error': True, 'reason': 'too_few_valid_frames', 'meta': meta_path}

    # Scale
    A_scaled = StandardScaler().fit_transform(A)
    S_scaled = StandardScaler().fit_transform(S)

    # Cap components to rank constraints
    k = int(max(1, min(n_components, A_scaled.shape[1], S_scaled.shape[1], len(A_scaled) - 1)))
    corrs = []
    p_perm = 1.0
    used_fallback = False
    try:
        cca = CCA(n_components=k, max_iter=500)
        A_c, S_c = cca.fit_transform(A_scaled, S_scaled)
        for i in range(k):
            ai = A_c[:, i]
            si = S_c[:, i]
            r = float(np.corrcoef(ai, si)[0, 1])
            corrs.append(r)
        corr1 = abs(corrs[0]) if corrs else 0.0
        rng = np.random.default_rng(123)
        greater = 0
        for _ in range(perm_iters):
            perm = rng.permutation(S_scaled)
            A_cp, S_cp = cca.fit_transform(A_scaled, perm)
            rp = float(np.corrcoef(A_cp[:, 0], S_cp[:, 0])[0, 1])
            if abs(rp) >= abs(corr1):
                greater += 1
        p_perm = float((greater + 1) / (perm_iters + 1))
    except Exception:
        # Fallback: pairwise max correlation across columns
        used_fallback = True
        R = np.corrcoef(np.hstack([A_scaled, S_scaled]).T)
        a_dim = A_scaled.shape[1]
        s_dim = S_scaled.shape[1]
        sub = R[:a_dim, a_dim:a_dim + s_dim]
        if sub.size > 0:
            max_idx = np.unravel_index(np.nanargmax(np.abs(sub)), sub.shape)
            corrs = [float(sub[max_idx])]
            # Permutation on rows
            rng = np.random.default_rng(123)
            greater = 0
            for _ in range(perm_iters):
                perm_rows = rng.permutation(len(A_scaled))
                r_perm = np.corrcoef(A_scaled[perm_rows, max_idx[0]], S_scaled[:, max_idx[1]])[0, 1]
                if abs(r_perm) >= abs(corrs[0]):
                    greater += 1
            p_perm = float((greater + 1) / (perm_iters + 1))

    return {
        'created_by': 'joe knowles',
        'timestamp': _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'),
        'file': meta['file'],
        'channel': meta['channel'],
        'wav_path': wav_path,
        'frames': int(len(A)),
        'win_s': win_s,
        'hop_s': hop_s,
        'n_mfcc': n_mfcc,
        'n_components': k,
        'canonical_correlations': corrs,
        'perm_p_first': p_perm,
        'fallback_pairwise': used_fallback,
    }


def write_summary(out_dir: str, entries: List[Dict]) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    js = os.path.join(out_dir, 'summary.json')
    with open(js, 'w') as f:
        json.dump(entries, f, indent=2)
    html = os.path.join(out_dir, 'summary.html')
    with open(html, 'w') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><title>MFCC+CCA Cross-Modal</title></head><body>\n')
        f.write('<h1>MFCC+CCA Cross-Modal Validation</h1>\n')
        f.write('<table border="1" cellspacing="0" cellpadding="6"><tr><th>File</th><th>Frames</th><th>CCA Corrs</th><th>Perm p (first)</th></tr>')
        for e in entries:
            if e.get('error'):
                f.write(f"<tr><td>{os.path.basename(str(e.get('file','')))}</td><td>-</td><td>ERROR</td><td>{e.get('reason','')}</td></tr>")
            else:
                corr_str = ', '.join(f"{c:.3f}" for c in e.get('canonical_correlations', [])[:3])
                f.write(f"<tr><td>{os.path.basename(e.get('file',''))}</td><td>{e.get('frames',0)}</td><td>{corr_str}</td><td>{e.get('perm_p_first',1.0):.4f}</td></tr>")
        f.write('</table></body></html>')
    return js, html


def main():
    import argparse
    ap = argparse.ArgumentParser(description='MFCC+CCA cross-modal validation over sonified runs')
    ap.add_argument('--runs_root', default=os.path.join('results', 'audio_continuous'))
    ap.add_argument('--out_root', default=os.path.join('results', 'cross_modal_mfcc_cca'))
    ap.add_argument('--win_s', type=float, default=1.0)
    ap.add_argument('--hop_s', type=float, default=0.5)
    ap.add_argument('--n_mfcc', type=int, default=20)
    ap.add_argument('--cca_components', type=int, default=3)
    ap.add_argument('--perm_iters', type=int, default=200)
    ap.add_argument('--force_basic_audio', action='store_true')
    args = ap.parse_args()

    metas = glob.glob(os.path.join(args.runs_root, '*', '*', 'metadata.json'))
    metas = sorted(metas)

    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    out_dir = os.path.join(args.out_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for mp in metas:
        try:
            res = per_run_cca(mp, args.win_s, args.hop_s, args.n_mfcc, args.cca_components, perm_iters=args.perm_iters, force_basic_audio=args.force_basic_audio)
        except Exception as e:
            res = {'error': True, 'reason': str(e), 'meta': mp}
        results.append(res)

    js, html = write_summary(out_dir, results)
    print(json.dumps({'summary_json': js, 'summary_html': html, 'count': len(results), 'out_dir': out_dir, 'metas': len(metas)}))


if __name__ == '__main__':
    main()


