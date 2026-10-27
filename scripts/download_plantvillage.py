"""Download and safely extract the unaugmented PlantVillage archive."""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
ARCHIVE = ROOT / "data" / "PlantVillage.zip"
PARTIAL = ROOT / "data" / "PlantVillage.zip.part"
DESTINATION = ROOT / "data" / "raw"
MAX_ATTEMPTS = 8


def readable_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def download_archive() -> bool:
    if ARCHIVE.exists():
        if readable_zip(ARCHIVE):
            return True
        print("The existing dataset archive is incomplete; trying to resume it.")
        if PARTIAL.exists():
            PARTIAL.unlink()
        ARCHIVE.replace(PARTIAL)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = PARTIAL.stat().st_size if PARTIAL.exists() else 0
        headers = {"User-Agent": "AgroShieldAI/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(URL, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    # The server ignored Range; restart instead of corrupting the archive.
                    offset = 0
                mode = "ab" if offset else "wb"
                content_length = int(response.headers.get("Content-Length", "0"))
                content_range = response.headers.get("Content-Range", "")
                total_size = content_length
                if "/" in content_range:
                    try:
                        total_size = int(content_range.rsplit("/", 1)[1])
                    except ValueError:
                        pass
                print(f"Downloading PlantVillage archive (attempt {attempt}/{MAX_ATTEMPTS})...")
                with PARTIAL.open(mode) as out:
                    copied = offset
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                        copied += len(chunk)
                        if total_size:
                            print(f"\rDownloaded {copied / total_size:.1%}", end="", flush=True)
                print()
            if readable_zip(PARTIAL):
                PARTIAL.replace(ARCHIVE)
                return True
            print("The server response is not a complete ZIP yet; retrying/resuming.")
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and readable_zip(PARTIAL):
                PARTIAL.replace(ARCHIVE)
                return True
            print(f"Download attempt {attempt} returned HTTP {exc.code}.", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"Download attempt {attempt} was interrupted: {exc}", file=sys.stderr)
        if attempt < MAX_ATTEMPTS:
            time.sleep(min(attempt * 2, 10))
    return False


def main() -> int:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not download_archive():
        print("Could not retrieve a complete ZIP archive.", file=sys.stderr)
        print(f"A partial download was kept at {PARTIAL} so the next run can attempt to resume it.", file=sys.stderr)
        print("Alternatively, download the unaugmented archive from https://data.mendeley.com/datasets/tywbtsjrjv/1 and save it as data/PlantVillage.zip.", file=sys.stderr)
        return 1

    DESTINATION.mkdir(parents=True, exist_ok=True)
    base = DESTINATION.resolve()
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            members = archive.infolist()
            for member in members:
                resolved = (DESTINATION / member.filename).resolve()
                if resolved != base and base not in resolved.parents:
                    raise ValueError(f"Unsafe path in archive: {member.filename}")
            archive.extractall(DESTINATION)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        print(f"Could not extract the archive: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset extracted under {DESTINATION}")
    print("Before training, inspect the downloaded dataset terms and cite the source in your report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
