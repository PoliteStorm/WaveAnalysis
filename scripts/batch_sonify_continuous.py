#!/usr/bin/env python3
import os
import json
import glob
import subprocess
import datetime as _dt

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'zenodo_5790768'))
OUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'audio_continuous'))


def list_species_files() -> list:
    # Accept .txt files; species names may include spaces
    return sorted(glob.glob(os.path.join(DATA_DIR, '*.txt')))


def run_single(path: str, speed: float = 3600.0, audio_fs: int = 22050,
               carrier: float = 660.0, depth: float = 0.9) -> dict:
    cmd = [
        'python3', os.path.join(os.path.dirname(__file__), 'sonify_continuous.py'),
        '--file', path,
        '--speed', str(speed),
        '--audio_fs', str(audio_fs),
        '--carrier', str(carrier),
        '--depth', str(depth),
        '--calibrate',
        '--out_dir', OUT_ROOT,
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # sonify_continuous prints JSON
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                return json.loads(line)
    except subprocess.CalledProcessError as e:
        return {'error': True, 'file': path, 'stdout': e.stdout, 'stderr': e.stderr}
    return {'error': True, 'file': path}


def write_index(entries: list):
    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    index_dir = os.path.join(OUT_ROOT, '_indexes')
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f'index_{ts}.json')
    with open(index_path, 'w') as f:
        json.dump(entries, f, indent=2)

    # Simple HTML index
    html_path = os.path.join(index_dir, f'index_{ts}.html')
    with open(html_path, 'w') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><title>Audio Index</title></head><body>\n')
        f.write('<h1>Continuous Sonification Index</h1>\n')
        for e in entries:
            if e.get('error'):
                f.write(f"<div><strong>ERROR</strong> {os.path.basename(e.get('file',''))}</div>\n")
                continue
            paths = e.get('paths', {})
            rel = os.path.relpath(paths.get('html', ''), start=index_dir)
            name = os.path.basename(e.get('file', ''))
            f.write(f"<div><a href='{rel}'>{name}</a></div>\n")
        f.write('</body></html>\n')
    return index_path, html_path


def main():
    paths = list_species_files()
    entries = []
    for p in paths:
        print(f"Rendering {p} ...", flush=True)
        meta = run_single(p)
        entries.append(meta)
    idx_json, idx_html = write_index(entries)
    print(json.dumps({'index_json': idx_json, 'index_html': idx_html, 'count': len(entries)}))


if __name__ == '__main__':
    main()


