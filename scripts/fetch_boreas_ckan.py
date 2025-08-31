#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import urllib.parse
import urllib.request
import http.cookiejar
import netrc as _netrc
from typing import Dict, List, Optional, Tuple


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "external")


def latest_manifest_path(root: str) -> Optional[str]:
    try:
        files = [
            f for f in os.listdir(root)
            if f.startswith("manifest_") and f.endswith(".json")
        ]
        if not files:
            return None
        files.sort()
        return os.path.join(root, files[-1])
    except Exception:
        return None


def read_manifest(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def write_manifest(path: str, manifest: Dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def find_entry(manifest: Dict, name: str) -> Optional[Dict]:
    for e in manifest.get("entries", []):
        if e.get("name") == name:
            return e
    return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ckan_package_show(url: str) -> Optional[Dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mushroooom/ckan-fetcher"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def guess_ckan_api_urls(package_id_or_url: str) -> List[str]:
    # If a full URL is provided, try to derive the API endpoint. Otherwise assume Data.gov CKAN
    parsed = urllib.parse.urlparse(package_id_or_url)
    if parsed.scheme and parsed.netloc:
        # Attempt common CKAN path replacement
        base = f"{parsed.scheme}://{parsed.netloc}"
        query_id = None
        qs = urllib.parse.parse_qs(parsed.query)
        if "id" in qs and qs["id"]:
            query_id = qs["id"][0]
        # Try last path segment as id as a fallback
        if not query_id:
            segs = [s for s in parsed.path.split("/") if s]
            if segs:
                query_id = segs[-1]
        if not query_id:
            return []
        return [
            f"{base}/api/3/action/package_show?id={urllib.parse.quote(query_id)}",
        ]
    else:
        # Treat as a package id; try Data.gov and catalog.data.gov
        pid = package_id_or_url
        return [
            f"https://catalog.data.gov/api/3/action/package_show?id={urllib.parse.quote(pid)}",
            f"https://data.gov/api/3/action/package_show?id={urllib.parse.quote(pid)}",
        ]


def select_resource_urls(pkg_json: Dict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    try:
        if not pkg_json.get("success"):
            return out
        result = pkg_json.get("result") or {}
        for r in result.get("resources", []) or []:
            url = r.get("url") or ""
            format_hint = (r.get("format") or "").lower()
            if not url:
                continue
            # Prefer CSV/ZIP direct files
            if any(url.lower().endswith(ext) for ext in (".csv", ".zip")) or format_hint in {"csv", "zip"}:
                name = r.get("name") or os.path.basename(urllib.parse.urlparse(url).path) or "resource"
                out.append((name, url))
    except Exception:
        pass
    return out


def build_earthdata_opener(use_netrc: bool, urs_host: str = "urs.earthdata.nasa.gov") -> Optional[urllib.request.OpenerDirector]:
    if not use_netrc:
        return None
    try:
        auth = _netrc.netrc()
        creds = auth.authenticators(urs_host)
        if not creds:
            return None
        login, account, password = creds
        pwd_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pwd_mgr.add_password(None, f"https://{urs_host}", login, password)
        auth_handler = urllib.request.HTTPBasicAuthHandler(pwd_mgr)
        cookie_jar = http.cookiejar.CookieJar()
        cookie_handler = urllib.request.HTTPCookieProcessor(cookie_jar)
        opener = urllib.request.build_opener(auth_handler, cookie_handler)
        return opener
    except Exception:
        return None


def download_file(url: str, out_dir: str, name_hint: str, earthdata_token: Optional[str] = None, use_netrc: bool = True) -> Optional[str]:
    ensure_dir(out_dir)
    try:
        parsed = urllib.parse.urlparse(url)
        fname = os.path.basename(parsed.path)
        if not fname:
            fname = name_hint.replace(" ", "_") + ".dat"
        out_path = os.path.join(out_dir, fname)
        headers = {"User-Agent": "mushroooom/ckan-fetcher"}
        # Add Bearer token for Earthdata-protected domains if provided
        if earthdata_token and parsed.netloc.endswith("earthdata.nasa.gov"):
            headers["Authorization"] = f"Bearer {earthdata_token}"
        req = urllib.request.Request(url, headers=headers)
        opener = None
        if parsed.netloc.endswith("earthdata.nasa.gov"):
            opener = build_earthdata_opener(use_netrc)
        if opener is None:
            opener = urllib.request.build_opener()
        with opener.open(req, timeout=90) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        return out_path
    except Exception:
        return None


def run(package_id_or_url: str, dataset_name: str, manifest_path: Optional[str], earthdata_token: Optional[str], use_netrc: bool) -> Dict:
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "external"))
    if not manifest_path:
        manifest_path = latest_manifest_path(data_root)
    if not manifest_path or not os.path.isfile(manifest_path):
        raise SystemExit("Manifest file not found; run scripts/fetch_external_datasets.py first.")
    manifest = read_manifest(manifest_path)
    entry = find_entry(manifest, dataset_name)
    if not entry:
        # Create a new entry
        entry = {
            "name": dataset_name,
            "url": package_id_or_url,
            "auth_required": False,
            "notes": "Added via CKAN second-pass fetcher",
            "downloaded": False,
            "local_paths": [],
        }
        manifest.setdefault("entries", []).append(entry)

    out_dir = os.path.join(data_root, dataset_name)
    ensure_dir(out_dir)

    api_urls = guess_ckan_api_urls(package_id_or_url)
    resources: List[Tuple[str, str]] = []
    for api_url in api_urls:
        pkg = ckan_package_show(api_url)
        if pkg:
            resources = select_resource_urls(pkg)
            if resources:
                break

    downloaded: List[str] = []
    for name_hint, url in resources:
        path = download_file(url, out_dir, name_hint, earthdata_token=earthdata_token, use_netrc=use_netrc)
        if path:
            # store as relative to repo root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            rel = os.path.relpath(path, start=repo_root)
            downloaded.append(rel)

    if downloaded:
        entry["downloaded"] = True
        # merge unique paths
        existing = set(entry.get("local_paths") or [])
        for p in downloaded:
            if p not in existing:
                entry.setdefault("local_paths", []).append(p)
                existing.add(p)

    write_manifest(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "downloaded_count": len(downloaded),
        "downloaded": downloaded,
        "resources_detected": len(resources),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Second-pass CKAN fetch for BOREAS dataset")
    ap.add_argument("--package", required=False, default="boreas-hyd-06-ground-gravimetric-soil-moisture-data-11935",
                    help="CKAN package id or full dataset URL")
    ap.add_argument("--dataset-name", required=False, default="BOREAS_HYD06_gravimetric",
                    help="Dataset name as used in external data directory and manifest")
    ap.add_argument("--manifest", required=False, default=None,
                    help="Path to existing external manifest JSON (defaults to latest)")
    ap.add_argument("--earthdata-token", required=False, default=os.getenv("EARTHDATA_TOKEN"),
                    help="Earthdata bearer token (or set env EARTHDATA_TOKEN)")
    ap.add_argument("--no-netrc", action="store_true", help="Do not use ~/.netrc for URS login")
    args = ap.parse_args()

    result = run(args.package, args.dataset_name, args.manifest, args.earthdata_token, (not args.no_netrc))
    print(json.dumps(result))


if __name__ == "__main__":
    main()


