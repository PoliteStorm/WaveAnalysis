#!/usr/bin/env python3
"""
Continuous sonification of long recordings suitable for Chromebook playback.

Features:
- Time scaling (speed-up) of entire recording
- AM carrier sonification (audible 440–880 Hz)
- Normalization and soft limiting
- WAV output + MP3 (via ffmpeg), and simple HTML player
"""

import os
import subprocess
import json
import datetime as _dt
from typing import Tuple

import numpy as np
from scipy.io import wavfile

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import prove_transform as pt


def robust_normalize(x: np.ndarray) -> np.ndarray:
    # Scale to ~[-1, 1] using 5-95th percentile
    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)
    rng = hi - lo
    if rng <= 1e-12:
        return np.zeros_like(x)
    z = (x - lo) / (rng + 1e-12)
    z = 2.0 * z - 1.0
    z = np.clip(z, -1.0, 1.0)
    return z


def soft_limiter(y: np.ndarray, drive: float = 2.0, out_level: float = 0.9) -> np.ndarray:
    # Simple tanh limiter
    if drive <= 0:
        drive = 1.0
    y_lim = np.tanh(drive * y) / np.tanh(drive)
    return out_level * y_lim.astype(np.float32)


def am_sonify_full(V: np.ndarray, fs: float, audio_fs: int, speed: float,
                   carrier_hz: float = 660.0, depth: float = 0.9,
                   calibrate: bool = True) -> np.ndarray:
    # Map audio time -> signal time
    dur_sig = len(V) / fs
    dur_audio = max(1e-3, dur_sig / max(speed, 1e-6))
    n_audio = int(dur_audio * audio_fs)
    t_audio = np.arange(n_audio, dtype=np.float32) / audio_fs
    t_sig = t_audio * speed

    # Interpolate voltage to audio timeline
    t_sig_axis = np.arange(len(V), dtype=np.float32) / fs
    V_interp = np.interp(t_sig, t_sig_axis, V).astype(np.float32)

    # Normalize and create AM modulator in [0,1]
    mod = robust_normalize(V_interp)
    mod = (1.0 - depth) + depth * (mod * 0.5 + 0.5)

    # Carrier
    carrier = np.sin(2 * np.pi * carrier_hz * t_audio).astype(np.float32)
    audio = (mod * carrier).astype(np.float32)

    # Optional 1s calibration tone at start
    if calibrate and n_audio > audio_fs:
        cal = 0.4 * np.sin(2 * np.pi * 440.0 * np.arange(audio_fs) / audio_fs).astype(np.float32)
        fade = int(0.05 * audio_fs)
        if fade > 0:
            w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(fade) / max(1, fade - 1)))
            cal[-fade:] *= (1 - w).astype(np.float32)
        audio[:audio_fs] += cal

    # Light high-pass (remove DC) using simple detrend
    audio = audio - np.mean(audio)

    # Soft limiter
    audio = soft_limiter(audio, drive=2.0, out_level=0.9)
    return audio


def write_outputs(audio: np.ndarray, audio_fs: int, out_dir: str) -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, 'continuous.wav')
    mp3_path = os.path.join(out_dir, 'continuous.mp3')
    html_path = os.path.join(out_dir, 'index.html')

    wavfile.write(wav_path, audio_fs, (audio * 32767.0).astype(np.int16))

    # Convert to MP3 using ffmpeg if available
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav_path,
                        '-ac', '1', '-ar', '22050', '-b:a', '128k', mp3_path], check=True)
    except Exception:
        mp3_path = ''

    # Simple HTML player
    with open(html_path, 'w') as f:
        f.write('<!doctype html>\n<html><head><meta charset="utf-8"><title>Continuous Sonification</title></head><body>\n')
        if mp3_path:
            f.write('<audio controls src="continuous.mp3"></audio>\n')
        else:
            f.write('<audio controls src="continuous.wav"></audio>\n')
        f.write('</body></html>\n')

    return wav_path, mp3_path, html_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Continuous AM sonification for Chromebook playback')
    ap.add_argument('--file', required=True, help='Zenodo TXT file')
    ap.add_argument('--fs', type=float, default=1.0)
    ap.add_argument('--audio_fs', type=int, default=22050)
    ap.add_argument('--carrier', type=float, default=660.0)
    ap.add_argument('--speed', type=float, default=3600.0, help='Time compression factor')
    ap.add_argument('--depth', type=float, default=0.9)
    ap.add_argument('--out_dir', default='results/audio_continuous')
    ap.add_argument('--calibrate', action='store_true')
    args = ap.parse_args()

    # Load
    t, channels = pt.load_zenodo_timeseries(args.file)
    pick = None
    for name, vec in channels.items():
        if np.isfinite(vec).any():
            pick = name
            break
    if pick is None:
        raise RuntimeError('No valid channel found')
    V = np.nan_to_num(channels[pick], nan=float(np.nanmean(channels[pick])))

    # Sonify
    audio = am_sonify_full(V, args.fs, args.audio_fs, args.speed, args.carrier, args.depth, calibrate=args.calibrate)

    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    species = os.path.splitext(os.path.basename(args.file))[0].replace(' ', '_')
    out_dir = os.path.join(args.out_dir, species, ts)

    wav_path, mp3_path, html_path = write_outputs(audio, args.audio_fs, out_dir)

    meta = {
        'created_by': 'joe knowles',
        'timestamp': ts,
        'file': args.file,
        'channel': pick,
        'fs_hz': args.fs,
        'audio_fs': args.audio_fs,
        'carrier_hz': args.carrier,
        'speed': args.speed,
        'depth': args.depth,
        'paths': {
            'wav': wav_path,
            'mp3': mp3_path,
            'html': html_path
        }
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta))


if __name__ == '__main__':
    main()


