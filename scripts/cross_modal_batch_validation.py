#!/usr/bin/env python3
import os
import json
import glob
import math
import datetime as _dt
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import prove_transform as pt


def sliding_frames(n: int, fs: int, win_s: float, hop_s: float) -> List[Tuple[int, int]]:
    win = int(win_s * fs)
    hop = int(hop_s * fs)
    if win <= 0:
        win = max(1, int(0.5 * fs))
    if hop <= 0:
        hop = max(1, int(0.25 * fs))
    frames = []
    start = 0
    while start + win <= n:
        frames.append((start, start + win))
        start += hop
    if not frames and n > 0:
        frames.append((0, n))
    return frames


def audio_features(x: np.ndarray, fs: int) -> Dict[str, float]:
    # Windowed features for a frame (already sliced)
    if len(x) == 0:
        return {
            'rms': 0.0, 'zcr': 0.0, 'centroid': 0.0,
            'bandwidth': 0.0, 'rolloff85': 0.0, 'flatness': 0.0
        }
    x = x.astype(np.float32)
    x = x - np.mean(x)
    # RMS
    rms = float(np.sqrt(np.mean(x * x)))
    # Zero crossing rate
    zc = np.where(np.diff(np.signbit(x)))[0]
    zcr = float(len(zc) / max(1, len(x) - 1))
    # Spectrum
    n = len(x)
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / max(1, n - 1)))
    X = np.fft.rfft(x * win)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = mag ** 2
    psd_sum = np.sum(psd) + 1e-12
    # Centroid
    centroid = float(np.sum(freqs * psd) / psd_sum)
    # Bandwidth (2nd moment)
    var = np.sum(((freqs - centroid) ** 2) * psd) / psd_sum
    bandwidth = float(np.sqrt(max(var, 0.0)))
    # Rolloff 85%
    cumsum = np.cumsum(psd)
    idx = int(np.searchsorted(cumsum, 0.85 * psd_sum))
    rolloff85 = float(freqs[min(idx, len(freqs) - 1)])
    # Flatness (geometric mean / arithmetic mean)
    gm = np.exp(np.mean(np.log(psd + 1e-12)))
    am = np.mean(psd + 1e-12)
    flatness = float(gm / am)
    return {
        'rms': rms, 'zcr': zcr, 'centroid': centroid,
        'bandwidth': bandwidth, 'rolloff85': rolloff85, 'flatness': flatness
    }


def signal_features_interp(V: np.ndarray, fs_signal: float, t0_sig: float, t1_sig: float, num_points: int = 2048) -> Dict[str, float]:
    # Interpolate signal onto uniform grid in [t0_sig, t1_sig]
    if t1_sig <= t0_sig:
        return {'mean': 0.0, 'std': 0.0, 'abs_deriv': 0.0, 'range': 0.0}
    t_axis = np.arange(len(V), dtype=np.float32) / float(fs_signal)
    t_grid = np.linspace(t0_sig, t1_sig, num_points, dtype=np.float32)
    v = np.interp(t_grid, t_axis, V).astype(np.float32)
    v = v - np.mean(v)
    std = float(np.std(v))
    abs_deriv = float(np.mean(np.abs(np.diff(v))))
    rng = float(np.max(v) - np.min(v))
    return {'mean': 0.0, 'std': std, 'abs_deriv': abs_deriv, 'range': rng}


def correlate_with_permutation(a: np.ndarray, b: np.ndarray, iters: int = 1000) -> Dict[str, float]:
    # Remove NaNs
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 5:
        return {'pearson_r': float('nan'), 'pearson_p': float('nan'), 'spearman_r': float('nan'), 'perm_p': float('nan')}
    r_p, p_p = pearsonr(a, b)
    r_s, _ = spearmanr(a, b)
    # Permutation p-value for absolute correlation
    observed = abs(r_p)
    count = 0
    rng = np.random.default_rng(12345)
    for _ in range(iters):
        perm = rng.permutation(b)
        r_perm, _ = pearsonr(a, perm)
        if abs(r_perm) >= observed:
            count += 1
    p_perm = (count + 1) / (iters + 1)
    return {'pearson_r': float(r_p), 'pearson_p': float(p_p), 'spearman_r': float(r_s), 'perm_p': float(p_perm)}


def process_run(meta_path: str, win_s: float = 1.0, hop_s: float = 0.5) -> Dict:
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    wav_path = meta['paths'].get('wav') or meta['paths'].get('mp3')  # prefer WAV
    if not wav_path or not os.path.exists(wav_path):
        return {'error': True, 'reason': 'wav not found', 'meta': meta_path}
    # Load audio
    fs_audio, audio = wavfile.read(wav_path)
    if audio.dtype.kind in ('i', 'u'):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Load signal
    t, channels = pt.load_zenodo_timeseries(meta['file'])
    V = np.asarray(channels[meta['channel']], dtype=np.float32)
    fs_signal = float(meta.get('fs_hz', 1.0))
    speed = float(meta.get('speed', 3600.0))

    # Timeline mapping: audio t -> signal t = t_audio * speed
    n_audio = len(audio)
    frames = sliding_frames(n_audio, fs_audio, win_s, hop_s)

    # Collect frame-wise features
    feats_audio = {k: [] for k in ['rms', 'zcr', 'centroid', 'bandwidth', 'rolloff85', 'flatness']}
    feats_signal = {k: [] for k in ['std', 'abs_deriv', 'range']}
    times_audio = []

    for s, e in frames:
        seg = audio[s:e]
        af = audio_features(seg, fs_audio)
        for k, v in af.items():
            feats_audio[k].append(v)
        # Map to signal interval
        t0_a = s / fs_audio
        t1_a = e / fs_audio
        t0_sig = t0_a * speed
        t1_sig = t1_a * speed
        sf = signal_features_interp(V, fs_signal, t0_sig, t1_sig)
        for k in ['std', 'abs_deriv', 'range']:
            feats_signal[k].append(sf[k])
        times_audio.append((t0_a + t1_a) * 0.5)

    # Convert to arrays
    for d in (feats_audio, feats_signal):
        for k in list(d.keys()):
            d[k] = np.asarray(d[k], dtype=np.float32)

    # Correlate pairs
    pairs = [
        ('rms', 'std'), ('rms', 'range'), ('centroid', 'abs_deriv'),
        ('bandwidth', 'range'), ('flatness', 'std'), ('rolloff85', 'abs_deriv')
    ]
    stats = {}
    for a, b in pairs:
        stats[f'{a}__{b}'] = correlate_with_permutation(feats_audio[a], feats_signal[b], iters=500)

    return {
        'created_by': 'joe knowles',
        'timestamp': _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'),
        'meta_path': meta_path,
        'wav_path': wav_path,
        'file': meta['file'],
        'channel': meta['channel'],
        'fs_hz': fs_signal,
        'audio_fs': fs_audio,
        'speed': speed,
        'window_s': win_s,
        'hop_s': hop_s,
        'stats': stats,
    }


def find_all_meta(root: str) -> List[str]:
    metas = glob.glob(os.path.join(root, '*', '*', 'metadata.json'))
    return sorted(metas)


def write_summary(out_dir: str, per_runs: List[Dict]) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    js_path = os.path.join(out_dir, 'summary.json')
    with open(js_path, 'w') as f:
        json.dump(per_runs, f, indent=2)

    # Minimal HTML summary
    html_path = os.path.join(out_dir, 'summary.html')
    with open(html_path, 'w') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><title>Cross-Modal Summary</title></head><body>\n')
        f.write('<h1>Cross-Modal Validation Summary</h1>\n')
        for run in per_runs:
            name = os.path.basename(run.get('file', ''))
            f.write(f"<h3>{name}</h3>\n")
            f.write('<pre>')
            f.write(json.dumps(run.get('stats', {}), indent=2))
            f.write('</pre>')
            wav = run.get('wav_path', '')
            if wav and os.path.exists(wav):
                rel = os.path.relpath(wav, start=out_dir)
                f.write(f"<audio controls src='{rel}'></audio>\n")
        f.write('</body></html>\n')
    return js_path, html_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Batch cross-modal validation for continuous sonifications')
    ap.add_argument('--runs_root', default=os.path.join('results', 'audio_continuous'))
    ap.add_argument('--out_root', default=os.path.join('results', 'cross_modal'))
    ap.add_argument('--win_s', type=float, default=1.0)
    ap.add_argument('--hop_s', type=float, default=0.5)
    args = ap.parse_args()

    meta_paths = find_all_meta(args.runs_root)
    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    out_dir = os.path.join(args.out_root, ts)

    per_runs = []
    for mp in meta_paths:
        try:
            res = process_run(mp, win_s=args.win_s, hop_s=args.hop_s)
        except Exception as e:
            res = {'error': True, 'meta_path': mp, 'exc': str(e)}
        per_runs.append(res)

    js_path, html_path = write_summary(out_dir, per_runs)
    print(json.dumps({'summary_json': js_path, 'summary_html': html_path, 'count': len(per_runs)}))


if __name__ == '__main__':
    main()


