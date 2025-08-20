#!/usr/bin/env python3
import os
import glob
from PIL import Image


def find_latest(dir_glob: str) -> str | None:
    runs = sorted(glob.glob(dir_glob))
    return runs[-1] if runs else None


def assemble_grid(panels: list[str], titles: list[str], out_path: str, cols: int = 2) -> str:
    images = [Image.open(p).convert('RGB') for p in panels if os.path.isfile(p)]
    if not images:
        raise SystemExit("No panels found to assemble")
    w = max(img.size[0] for img in images)
    h = max(img.size[1] for img in images)
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    W = cols * w
    H = rows * h
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        x = c * w
        y = r * h
        if img.size != (w, h):
            img = img.resize((w, h))
        canvas.paste(img, (x, y))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, format='PNG')
    return out_path


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'zenodo'))
    species = [d for d in sorted(glob.glob(os.path.join(base, '*'))) if os.path.isdir(d)]
    panels = []
    titles = []
    for sd in species:
        latest = find_latest(os.path.join(sd, '*'))
        if not latest:
            continue
        p = os.path.join(latest, 'summary_panel.png')
        if os.path.isfile(p):
            panels.append(p)
            titles.append(os.path.basename(sd))
    if not panels:
        raise SystemExit("No summary_panel.png found")
    out_dir = os.path.join(base, '_composites')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'species_gallery.png')
    assemble_grid(panels, titles, out_path, cols=2)
    print(out_path)


if __name__ == '__main__':
    main()


