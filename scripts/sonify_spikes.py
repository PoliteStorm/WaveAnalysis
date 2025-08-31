#!/usr/bin/env python3
"""
Sonify spikes into an audible WAV using a Fibonacci-based pitch mapping.
Also performs simple cross-modal validation by correlating audio envelope
with original voltage envelope.
"""

import os
import json
import math
import datetime as _dt
from typing import List, Dict, Tuple

import numpy as np
from scipy.io import wavfile

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import prove_transform as pt


def simple_spike_detect(v_mV: np.ndarray, fs_hz: float, min_amp_mV: float = 0.2,
                        min_isi_s: float = 30.0, baseline_win_s: float = 600.0) -> List[Dict]:
    w = max(1, int(round(baseline_win_s * fs_hz)))
    if w >= len(v_mV):
        baseline = np.full_like(v_mV, np.mean(v_mV))
    else:
        kernel = np.ones(w, dtype=float) / w
        baseline = np.convolve(v_mV, kernel, mode='same')
        baseline[:w//2] = baseline[w//2]
        baseline[-(w//2):] = baseline[-(w//2)]
    x = v_mV - baseline
    above = np.where(np.abs(x) >= min_amp_mV)[0]
    events: List[Tuple[int, int]] = []
    if above.size == 0:
        return []
    start = above[0]
    prev = above[0]
    for idx in above[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            events.append((start, prev))
            start = idx
            prev = idx
    events.append((start, prev))
    min_gap = int(round(min_isi_s * fs_hz))
    merged: List[Tuple[int, int]] = []
    last_end = -10**9
    for a, b in events:
        if a - last_end < min_gap and merged:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
        last_end = merged[-1][1]
    spikes: List[Dict] = []
    for a, b in merged:
        seg = v_mV[a:b+1]
        if seg.size == 0:
            continue
        peak_idx_local = int(np.argmax(np.abs(seg)))
        peak_idx = a + peak_idx_local
        spikes.append({
            't_idx': int(peak_idx),
            't_s': float(peak_idx / fs_hz),
            'amplitude_mV': float(v_mV[peak_idx])
        })
    return spikes


def fib_scale(n: int) -> List[int]:
    a, b = 1, 1
    seq = [a, b]
    while len(seq) < n:
        a, b = b, a + b
        seq.append(b)
    return seq[:n]


def render_sonification(spike_times_s: np.ndarray, spike_amps: np.ndarray,
                        fs_audio: int = 44100, duration_s: float = 120.0,
                        base_freq: float = 220.0, tone_ms: float = 250.0,
                        n_fib: int = 8) -> np.ndarray:
    n_samples = int(duration_s * fs_audio)
    audio = np.zeros(n_samples, dtype=np.float32)
    # Fibonacci scale ratios normalized
    fib = np.array(fib_scale(n_fib), dtype=float)
    fib = fib / fib[0]
    # Map amplitudes to indices
    if spike_amps.size > 0:
        amp_norm = (np.abs(spike_amps) - np.min(np.abs(spike_amps))) / (np.ptp(np.abs(spike_amps)) + 1e-12)
    else:
        amp_norm = np.array([])
    for i, t_s in enumerate(spike_times_s):
        onset = int(t_s * fs_audio)
        if onset >= n_samples:
            continue
        dur = int((tone_ms / 1000.0) * fs_audio)
        end = min(n_samples, onset + dur)
        idx = int(np.floor((amp_norm[i] if amp_norm.size else 0.0) * (n_fib - 1)))
        freq = base_freq * fib[idx]
        # Sine burst with Hann window
        tt = np.arange(end - onset, dtype=float) / fs_audio
        phase = 2 * np.pi * freq * tt
        burst = np.sin(phase).astype(np.float32)
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(burst.size) / max(1, burst.size - 1))).astype(np.float32)
        burst *= w
        audio[onset:end] += burst
    # Normalize
    max_abs = np.max(np.abs(audio)) + 1e-9
    audio = 0.9 * (audio / max_abs)
    return audio


def envelope(signal: np.ndarray, win: int) -> np.ndarray:
    win = max(1, win)
    kernel = np.ones(win, dtype=float) / win
    env = np.convolve(np.abs(signal), kernel, mode='same')
    return env


def cross_modal_validation(audio: np.ndarray, fs_audio: int,
                           v_mV: np.ndarray, fs_hz: float) -> Dict:
    # Audio envelope at ~10 Hz
    env_audio = envelope(audio, max(1, int(fs_audio / 10)))
    # Downsample audio envelope to 1 Hz to match voltage
    n_secs = int(len(audio) / fs_audio)
    env_audio_1hz = []
    for s in range(n_secs):
        seg = env_audio[s * fs_audio:(s + 1) * fs_audio]
        if seg.size:
            env_audio_1hz.append(float(np.mean(seg)))
    env_audio_1hz = np.array(env_audio_1hz, dtype=float)
    # Voltage envelope (1 Hz)
    v_env = envelope(v_mV, 60)  # 1-minute smooth
    # Equalize lengths
    m = min(len(env_audio_1hz), len(v_env))
    if m < 5:
        return {'status': 'insufficient_overlap'}
    a = env_audio_1hz[:m]
    b = v_env[:m]
    # Correlation
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
    return {
        'status': 'ok',
        'corr_audio_voltage_envelope': corr,
        'overlap_secs': int(m)
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Sonify spikes with Fibonacci mapping and validate')
    ap.add_argument('--file', required=True, help='Zenodo TXT file')
    ap.add_argument('--channel', default='', help='Optional channel')
    ap.add_argument('--fs', type=float, default=1.0)
    ap.add_argument('--out_dir', default='results/audio')
    ap.add_argument('--audio_fs', type=int, default=44100)
    ap.add_argument('--duration', type=float, default=120.0, help='Audio duration (s)')
    ap.add_argument('--compress', action='store_true', help='Time-compress spike times to fill audio duration')
    ap.add_argument('--calibrate', action='store_true', help='Prepend a 1s 440Hz calibration tone to ensure audibility')
    args = ap.parse_args()

    # Load data
    t, channels = pt.load_zenodo_timeseries(args.file)
    pick = args.channel if args.channel and args.channel in channels else None
    if pick is None:
        for name, vec in channels.items():
            if np.isfinite(vec).any():
                pick = name
                break
    V = np.nan_to_num(channels[pick], nan=float(np.nanmean(channels[pick])))

    # Detect spikes
    spikes = simple_spike_detect(V, args.fs)
    spike_times = np.array([s['t_s'] for s in spikes], dtype=float)
    spike_amps = np.array([s['amplitude_mV'] for s in spikes], dtype=float)

    # Determine time window
    max_t = args.duration

    # If compress is requested and we have spikes outside the window, rescale to [0, duration]
    if args.compress and spike_times.size > 0:
        t0 = float(np.min(spike_times))
        t1 = float(np.max(spike_times))
        if t1 > t0:
            spike_times = (spike_times - t0) / (t1 - t0) * max_t
        else:
            spike_times = np.zeros_like(spike_times)
    else:
        # Otherwise keep only spikes within [0, duration]
        mask = spike_times <= max_t
        spike_times = spike_times[mask]
        spike_amps = spike_amps[mask] if spike_amps.size else spike_amps

    # Fallback: if no spikes available for audio, synthesize peaks from voltage envelope
    if spike_times.size == 0:
        v_abs = np.abs(V)
        win = max(1, int(round(60 * args.fs)))
        kernel = np.ones(win, dtype=float) / win
        v_env = np.convolve(v_abs, kernel, mode='same')
        # pick top N peaks uniformly across
        N = 32
        idx = np.argsort(v_env)[-N:]
        idx.sort()
        tt = idx / args.fs
        if args.compress and tt.size > 0:
            t0 = float(np.min(tt))
            t1 = float(np.max(tt))
            tt = (tt - t0) / max(1e-9, (t1 - t0)) * max_t
        else:
            tt = tt[tt <= max_t]
        spike_times = tt.astype(float)
        spike_amps = v_env[idx] if v_env.size else np.ones_like(tt)

    # Render audio
    audio = render_sonification(spike_times, spike_amps, fs_audio=args.audio_fs, duration_s=max_t)

    # Optional calibration tone at start (1s @ 440Hz)
    if args.calibrate:
        cal_len = min(args.audio_fs, audio.shape[0])
        tt = np.arange(cal_len, dtype=float) / args.audio_fs
        cal = 0.5 * np.sin(2*np.pi*440.0*tt).astype(np.float32)
        # 50ms fade out to avoid click
        fade = int(0.05 * args.audio_fs)
        if fade > 0 and cal_len > fade:
            w = 0.5 * (1 - np.cos(2*np.pi*np.arange(fade)/max(1,fade-1)))
            cal[-fade:] *= (1 - w).astype(np.float32)
        audio[:cal_len] += cal

    # If audio is extremely quiet, add marker beeps every 5s @ 880Hz
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    if rms < 1e-3:
        step = 5 * args.audio_fs
        beep_len = int(0.2 * args.audio_fs)
        for start in range(0, len(audio), step):
            end = min(len(audio), start + beep_len)
            tt = np.arange(end - start, dtype=float) / args.audio_fs
            audio[start:end] += 0.4 * np.sin(2*np.pi*880.0*tt).astype(np.float32)
    audio_i16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    # Output paths
    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    species = os.path.splitext(os.path.basename(args.file))[0].replace(' ', '_')
    out_dir = os.path.join(args.out_dir, species, ts)
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, 'sonification.wav')
    json_map = os.path.join(out_dir, 'sonification_mapping.json')
    json_val = os.path.join(out_dir, 'cross_modal_validation.json')

    wavfile.write(wav_path, args.audio_fs, audio_i16)

    mapping = {
        'created_by': 'joe knowles',
        'timestamp': ts,
        'file': args.file,
        'channel': pick,
        'audio_fs': args.audio_fs,
        'duration_s': max_t,
        'mapping': 'spike->sine burst, amplitude->Fibonacci index, 120ms Hann-windowed tones',
        'fib_sequence': fib_scale(8),
        'calibration_tone': bool(args.calibrate)
    }
    with open(json_map, 'w') as f:
        json.dump(mapping, f, indent=2)

    # Cross-modal validation
    val = cross_modal_validation(audio, args.audio_fs, V[:int(min(max_t, len(V)/args.fs) * args.fs)], args.fs)
    with open(json_val, 'w') as f:
        json.dump(val, f, indent=2)

    print(json.dumps({'wav': wav_path, 'mapping': json_map, 'validation': json_val}))


if __name__ == '__main__':
    main()


