#!/usr/bin/env python3
import os
import glob
import shutil


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_if_exists(src: str, dst: str) -> None:
    if os.path.isfile(src):
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)


def latest_run(dir_glob: str) -> str | None:
    runs = sorted(glob.glob(dir_glob))
    return runs[-1] if runs else None


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    figs_dir = os.path.join(root, 'docs', 'paper', 'figs')
    ensure_dir(figs_dir)

    # Species figures
    zroot = os.path.join(root, 'results', 'zenodo')
    for sp_dir in sorted(glob.glob(os.path.join(zroot, '*'))):
        if not os.path.isdir(sp_dir) or os.path.basename(sp_dir) == '_composites':
            continue
        latest = latest_run(os.path.join(sp_dir, '*'))
        if not latest:
            continue
        sp = os.path.basename(sp_dir)
        copy_if_exists(os.path.join(latest, 'summary_panel.png'), os.path.join(figs_dir, f'{sp}_summary.png'))
        copy_if_exists(os.path.join(latest, 'tau_band_power_heatmap.png'), os.path.join(figs_dir, f'{sp}_heatmap.png'))
        copy_if_exists(os.path.join(latest, 'tau_band_power_surface.png'), os.path.join(figs_dir, f'{sp}_surface.png'))
        copy_if_exists(os.path.join(latest, 'hist_isi.png'), os.path.join(figs_dir, f'{sp}_hist_isi.png'))
        copy_if_exists(os.path.join(latest, 'hist_amp.png'), os.path.join(figs_dir, f'{sp}_hist_amp.png'))

    # ML figures
    ml_root = os.path.join(root, 'results', 'ml')
    latest_ml = latest_run(os.path.join(ml_root, '*'))
    if latest_ml:
        copy_if_exists(os.path.join(latest_ml, 'figs', 'feature_importance.png'), os.path.join(figs_dir, 'ml_feature_importance.png'))
        copy_if_exists(os.path.join(latest_ml, 'figs', 'confusion_matrix.png'), os.path.join(figs_dir, 'ml_confusion_matrix.png'))
        copy_if_exists(os.path.join(latest_ml, 'figs', 'calibration.png'), os.path.join(figs_dir, 'ml_calibration.png'))

    print(figs_dir)


if __name__ == '__main__':
    main()


