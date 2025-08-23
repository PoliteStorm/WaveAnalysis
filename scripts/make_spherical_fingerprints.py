#!/usr/bin/env python3
import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import plotly.graph_objects as go


def latest_run_dir(species_dir: str) -> str | None:
    runs = sorted(glob.glob(os.path.join(species_dir, '*')))
    return runs[-1] if runs else None


def load_json(path: str) -> dict | None:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def build_sphere(band_fracs: dict, taus_sorted: list[float], ci_halfwidths: list[float] | None,
                 spike_count: int, ampl_entropy_bits: float | None,
                 conc_sqrt: float, snr_ratio: float):
    # Sphere mesh (higher resolution for clarity)
    phi = np.linspace(0, np.pi, 120)   # latitude (0..pi)
    theta = np.linspace(0, 2*np.pi, 240)  # longitude
    PHI, THETA = np.meshgrid(phi, theta)
    r_base = 1.0
    X = r_base * np.sin(PHI) * np.cos(THETA)
    Y = r_base * np.sin(PHI) * np.sin(THETA)
    Z = r_base * np.cos(PHI)

    # Map τ to bands at specific latitudes (equator=fast → poles=slow)
    latitudes = np.linspace(np.pi/2 - 0.6, np.pi/2 + 0.6, len(taus_sorted))
    fracs = np.array([float(band_fracs.get(str(t), band_fracs.get(t, 0.0))) for t in taus_sorted], dtype=float)
    if np.sum(fracs) > 0:
        fracs = fracs / np.sum(fracs)
    ci = np.array(ci_halfwidths, dtype=float) if ci_halfwidths is not None else np.zeros_like(fracs)

    # Color bands by fraction; for each grid row choose nearest τ-band
    C = np.zeros_like(X)
    tau_idx_grid = np.zeros_like(X, dtype=int)
    for j in range(C.shape[1]):
        nearest = np.argmin(np.abs(PHI[:, j][:, None] - latitudes[None, :]), axis=1)
        C[:, j] = fracs[nearest]
        tau_idx_grid[:, j] = nearest

    # Add a scalar bump from conc+snr toward +Z
    bump = 0.15 * (np.clip(conc_sqrt, 0.0, 1.0) + 0.5 * np.clip(snr_ratio, 0.0, 2.0))
    Zb = Z + bump * (Z / (np.max(np.abs(Z)) + 1e-9))

    return X, Y, Z, Zb, C, latitudes, fracs, ci, tau_idx_grid


def main():
    ap = argparse.ArgumentParser(description='Generate interactive spherical fingerprints per species')
    ap.add_argument('--results_root', default='results/zenodo')
    ap.add_argument('--out_root', default='results/fingerprints')
    args = ap.parse_args()

    species = [
        'Schizophyllum_commune',
        'Enoki_fungi_Flammulina_velutipes',
        'Ghost_Fungi_Omphalotus_nidiformis',
        'Cordyceps_militari',
    ]

    ts = datetime.now().isoformat()
    for sp in species:
        sp_dir = os.path.join(args.results_root, sp)
        run_dir = latest_run_dir(sp_dir)
        if not run_dir:
            print(f"[SKIP] {sp}: no runs")
            continue
        metrics = load_json(os.path.join(run_dir, 'metrics.json'))
        conc = load_json(os.path.join(run_dir, 'snr_concentration.json'))
        if not metrics or not conc:
            print(f"[SKIP] {sp}: missing metrics or concentration")
            continue
        # τ order
        band_fracs = metrics.get('band_fractions', {})
        taus_sorted = sorted([float(k) for k in band_fracs.keys()])
        # Try CI from ci_summaries if present
        ci_dir = os.path.join('results', 'ci_summaries', sp)
        ci_runs = sorted(glob.glob(os.path.join(ci_dir, '*')))
        ci_halfwidths = None
        if ci_runs:
            ci_json = os.path.join(ci_runs[-1], 'tau_power_ci.json')
            ci_data = load_json(ci_json)
            if ci_data and ci_data.get('taus'):
                # align order to taus_sorted; tolerate labels like 'tau_5.5'
                import re
                def parse_tau(v):
                    if isinstance(v, (int, float)):
                        return float(v)
                    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(v))
                    return float(m.group(0)) if m else float('nan')
                taus_ci = [parse_tau(x) for x in ci_data['taus']]
                tau_to_idx = {float(t): i for i, t in enumerate(taus_ci)}
                lo = np.array(ci_data['lo'], dtype=float)
                hi = np.array(ci_data['hi'], dtype=float)
                hw = (hi - lo) / 2.0
                ci_halfwidths = [float(hw[tau_to_idx.get(t, 0)]) for t in taus_sorted]

        spike_count = int(metrics.get('spike_count', 0) or 0)
        ampl_entropy = float(metrics.get('amplitude_stats', {}).get('shannon_entropy_bits', 0.0))
        snr_sqrt = float(conc.get('snr', {}).get('sqrt', 0.0))
        snr_stft = float(conc.get('snr', {}).get('stft', 1.0))
        conc_sqrt = float(conc.get('concentration', {}).get('sqrt', 0.0))
        snr_ratio = (snr_sqrt + 1e-12) / (snr_stft + 1e-12)

        X, Y, Z0, Zb, C, lats, fracs, ci, tau_idx_grid = build_sphere(band_fracs, taus_sorted, ci_halfwidths, spike_count, ampl_entropy, conc_sqrt, snr_ratio)

        # Build customdata for hover: tau, fraction
        tau_vals_grid = np.vectorize(lambda idx: taus_sorted[int(idx)])(tau_idx_grid)
        custom = np.dstack((tau_vals_grid, C))

        # Bumped surface with lighting and informative hover
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Zb, surfacecolor=C, colorscale='Viridis', showscale=True,
            cmin=0.0, cmax=1.0, colorbar=dict(title='τ fraction'),
            customdata=custom,
            hovertemplate='τ=%{customdata[0]:.3g}<br>fraction=%{customdata[1]:.3f}<extra></extra>',
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.3, roughness=0.45),
            lightposition=dict(x=100, y=200, z=300),
        )])

        # Add latitude rings (center; CI as thickness via inner/outer rings if available)
        rings = []
        max_ci = float(np.max(ci)) if ci.size > 0 else 0.0
        for i, lat in enumerate(lats):
            long = np.linspace(0, 2*np.pi, 360)
            # center ring
            xring = np.sin(lat) * np.cos(long)
            yring = np.sin(lat) * np.sin(long)
            zring = np.cos(lat) + 0.001
            rings.append(go.Scatter3d(x=xring, y=yring, z=zring, mode='lines',
                                      line=dict(color='white', width=2), opacity=1.0,
                                      name=f"τ={taus_sorted[i]:g}", showlegend=False))
            # CI thickness rings
            if max_ci > 0 and i < len(ci):
                offset = 0.2 * (ci[i] / (max_ci + 1e-9))
                for sign in (+1, -1):
                    lat_use = lat + sign * offset
                    xri = np.sin(lat_use) * np.cos(long)
                    yri = np.sin(lat_use) * np.sin(long)
                    zri = np.cos(lat_use) + 0.001
                    rings.append(go.Scatter3d(x=xri, y=yri, z=zri, mode='lines',
                                              line=dict(color='white', width=1), opacity=0.5,
                                              showlegend=False))
        for tr in rings:
            fig.add_trace(tr)

        # Add 3D text labels at each latitude band using a Scatter3d trace
        xs, ys, zs, texts = [], [], [], []
        for i, lat in enumerate(lats):
            xs.append(float(np.sin(lat)))
            ys.append(0.0)
            zs.append(float(np.cos(lat)) + 0.02)
            texts.append(f"τ={taus_sorted[i]:g}, f={fracs[i]:.2f}")
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='text', text=texts, textposition='top center', name='τ bands', showlegend=False))
        fig.update_layout(
            title=f"{sp.replace('_',' ')} — spherical fingerprint",
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        )

        out_dir = os.path.join(args.out_root, sp, ts)
        os.makedirs(out_dir, exist_ok=True)
        out_html = os.path.join(out_dir, 'sphere.html')
        fig.write_html(out_html, include_plotlyjs='cdn', full_html=True, config={'responsive': True, 'displaylogo': False})
        # write a mapping JSON
        mapping = {
            'created_by': 'joe knowles',
            'timestamp': ts,
            'species': sp,
            'encodings': {
                'latitude_bands': 'τ (fast→equator, slow→poles)',
                'surfacecolor': 'mean τ-fraction (unitless, sum≈1)',
                'surface bump (z)': '√t concentration + SNR contrast',
                'annotations': 'τ and fraction at latitude bands',
            },
            'taus_sorted': taus_sorted,
            'fractions': [float(x) for x in fracs],
            'ci_halfwidths': [float(x) for x in ci] if ci_halfwidths is not None else None,
            'snr': {'sqrt': snr_sqrt, 'stft': snr_stft},
            'concentration': {'sqrt': conc_sqrt},
            'references': [
                'Sci Rep 2018: https://www.nature.com/articles/s41598-018-26007-1',
                'Jones 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/',
                'Adamatzky 2022: https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/',
                'Biosystems 2021: https://www.sciencedirect.com/science/article/pii/S0303264721000307',
            ],
        }
        with open(os.path.join(out_dir, 'sphere.json'), 'w') as f:
            json.dump(mapping, f, indent=2)
        # write references.md for consistency
        refs_md = os.path.join(out_dir, 'references.md')
        with open(refs_md, 'w') as rf:
            rf.write("- Spiking in Pleurotus djamor (Sci Rep 2018): https://www.nature.com/articles/s41598-018-26007-1\n")
            rf.write("- Multiscalar electrical spiking in Schizophyllum commune (Sci Rep 2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/\n")
            rf.write("- Language of fungi derived from electrical spiking (R. Soc. Open Sci. 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC8984380/\n")
            rf.write("- Electrical activity of fungi: Spikes detection and complexity (Biosystems 2021): https://www.sciencedirect.com/science/article/pii/S0303264721000307\n")
        print(f"[OK] {out_html}")


if __name__ == '__main__':
    main()


