#!/usr/bin/env python3
import os
import sys
import json
import time
import pathlib
import urllib.request

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'external')

SOURCES = [
    {
        'name': 'FIFE_soil_moisture_gravimetric',
        'url': 'https://daac.ornl.gov/FIFE/guides/Soil_Moisture_Gravimetric_Data.html',
        'notes': 'Requires Earthdata sign-in for bulk files; this link documents access. Use wget with Earthdata netrc.',
        'auth_required': True,
    },
    {
        'name': 'BOREAS_HYD06_gravimetric',
        'url': 'https://catalog.data.gov/dataset/boreas-hyd-06-ground-gravimetric-soil-moisture-data-11935',
        'notes': 'Landing page; follow resource links to CSV/ZIP files.',
        'auth_required': False,
    },
    {
        'name': 'KBS_LTER_gravimetric',
        'url': 'https://lter.kbs.msu.edu/datatables/30',
        'notes': 'Landing page with CSV export; manual or scripted CSV export may require form params.',
        'auth_required': False,
    },
    {
        'name': 'OpenDataBay_env_conditions',
        'url': 'https://www.opendatabay.com/data/ai-ml/93f63166-fd70-48bf-8067-9147c717fd41',
        'notes': 'Direct download may require browser; record landing page and instructions.',
        'auth_required': False,
    },
]

MANIFEST = []


def ensure_dirs():
    os.makedirs(DATA_ROOT, exist_ok=True)
    for s in SOURCES:
        os.makedirs(os.path.join(DATA_ROOT, s['name']), exist_ok=True)


def try_simple_download(name: str, url: str) -> str | None:
    # Attempt to download if the URL seems to be a direct file
    out_dir = os.path.join(DATA_ROOT, name)
    fname = os.path.basename(urllib.parse.urlparse(url).path)
    if not fname or '.' not in fname:
        return None
    out_path = os.path.join(out_dir, fname)
    try:
        urllib.request.urlretrieve(url, out_path)
        return out_path
    except Exception:
        return None


def main():
    ensure_dirs()
    ts = time.strftime('%Y-%m-%dT%H-%M-%S')
    for s in SOURCES:
        entry = {
            'name': s['name'],
            'url': s['url'],
            'auth_required': s['auth_required'],
            'notes': s['notes'],
            'downloaded': False,
            'local_paths': [],
        }
        path = try_simple_download(s['name'], s['url'])
        if path:
            entry['downloaded'] = True
            entry['local_paths'].append(os.path.abspath(path))
        MANIFEST.append(entry)
    manifest_path = os.path.join(DATA_ROOT, f'manifest_{ts}.json')
    with open(manifest_path, 'w') as f:
        json.dump({'created_by': 'joe knowles', 'timestamp': ts, 'entries': MANIFEST}, f, indent=2)
    print(json.dumps({'manifest': manifest_path, 'entries': len(MANIFEST)}))


if __name__ == '__main__':
    main()
