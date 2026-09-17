"""Download and verify the Geo-VAE inference weights from GitHub Releases."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def verified(path, info):
    if not path.is_file() or path.stat().st_size != info["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == info["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repository as OWNER/NAME")
    parser.add_argument("--tag", default="review-v1")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "checkpoints")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("--repo must be OWNER/NAME")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("--tag must contain only letters, digits, dots, underscores, or hyphens")
    manifest_path = Path(__file__).resolve().parents[1] / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, info in manifest.items():
        target = args.output / name
        if verified(target, info):
            print(f"Already verified: {name}")
            continue
        partial = target.with_name(target.name + ".part")
        url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{name}"
        print(f"Downloading {name} ({info['bytes']:,} bytes)", flush=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GeoVAE-weight-download"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            if not verified(partial, info):
                raise RuntimeError(f"Size or SHA-256 verification failed for {name}")
            partial.replace(target)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        print(f"Verified: {name}")


if __name__ == "__main__":
    main()
