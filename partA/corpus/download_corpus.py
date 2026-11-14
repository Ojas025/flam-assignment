#!/usr/bin/env python3
"""Download and extract the FLORES-200 and MASSIVE raw corpora."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

FLORES_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
MASSIVE_DOWNLOAD_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
MASSIVE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
DOWNLOAD_RETRIES = 3
RETRY_BACKOFF_S = 2.0
CHUNK_SIZE = 1 << 20
CONNECT_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 60.0


class DownloadError(Exception):
    """Raised when a corpus archive cannot be downloaded or validated."""


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    url: str
    sha256: str
    archive_name: str
    extract_dir_name: str


SPECS: dict[str, CorpusSpec] = {
    "flores": CorpusSpec(
        name="flores",
        url=FLORES_DOWNLOAD_URL,
        sha256=FLORES_SHA256,
        archive_name="flores200_dataset.tar.gz",
        extract_dir_name="flores200_dataset",
    ),
    "massive": CorpusSpec(
        name="massive",
        url=MASSIVE_DOWNLOAD_URL,
        sha256=MASSIVE_SHA256,
        archive_name="amazon-massive-dataset-1.1.tar.gz",
        extract_dir_name="1.1",
    ),
}


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(spec: CorpusSpec, archive: Path) -> None:
    """Validate archive SHA-256 and tar readability."""
    actual = sha256_of(archive)
    if actual != spec.sha256:
        raise DownloadError(
            f"{archive.name}: SHA-256 mismatch (expected {spec.sha256}, got {actual})"
        )
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.getmembers()
    except tarfile.TarError as exc:
        raise DownloadError(f"{archive.name}: not a readable tar.gz archive ({exc})") from exc


def download(spec: CorpusSpec, archive: Path) -> None:
    """Stream the archive to a .part file, validate it, then rename into place."""
    part = archive.with_suffix(archive.suffix + ".part")
    try:
        with httpx.stream(
            "GET",
            spec.url,
            follow_redirects=True,
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        ) as response:
            response.raise_for_status()
            expected = response.headers.get("content-length")
            written = 0
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    handle.write(chunk)
                    written += len(chunk)
        if expected is not None and written != int(expected):
            raise DownloadError(
                f"{spec.name}: truncated download ({written} of {expected} bytes)"
            )
        verify(spec, part)
        part.replace(archive)
    finally:
        part.unlink(missing_ok=True)


def download_with_retries(spec: CorpusSpec, archive: Path) -> None:
    """Download with bounded retries and backoff on network errors."""
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            download(spec, archive)
            return
        except (httpx.HTTPStatusError, httpx.RequestError, DownloadError) as exc:
            last_error = exc
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    raise DownloadError(
        f"{spec.name}: download failed after {DOWNLOAD_RETRIES} attempts: {last_error}"
    ) from last_error


def extract(spec: CorpusSpec, archive: Path, dest: Path) -> None:
    """Extract into a temp dir, then rename into place so partial extracts never count."""
    final = dest / spec.extract_dir_name
    base = dest / f".{spec.extract_dir_name}.tmp"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(base, filter="data")
    except tarfile.TarError as exc:
        shutil.rmtree(base, ignore_errors=True)
        raise DownloadError(f"{archive.name}: extraction failed ({exc})") from exc
    except OSError as exc:
        shutil.rmtree(base, ignore_errors=True)
        raise DownloadError(f"{archive.name}: cannot write to {dest} ({exc})") from exc
    root = base / spec.extract_dir_name
    source = root if root.is_dir() else base
    if final.exists():
        shutil.rmtree(final)
    source.rename(final)
    if source is not base:
        shutil.rmtree(base, ignore_errors=True)


def process_corpus(spec: CorpusSpec, dest: Path, force: bool, no_extract: bool) -> None:
    """Download, verify, and optionally extract one corpus."""
    target = dest / spec.name
    target.mkdir(parents=True, exist_ok=True)
    archive = target / spec.archive_name
    extracted = target / spec.extract_dir_name

    if archive.exists() and not force:
        try:
            verify(spec, archive)
            print(f"{spec.name}: archive present and valid, skipping download")
        except DownloadError as exc:
            print(f"{spec.name}: existing archive invalid ({exc}), re-downloading")
            archive.unlink()
            download_with_retries(spec, archive)
    else:
        download_with_retries(spec, archive)
    print(f"{spec.name}: archive verified at {archive}")

    if no_extract:
        return
    if extracted.exists() and not force:
        print(f"{spec.name}: {extracted} already extracted, skipping")
        return
    extract(spec, archive, target)
    print(f"{spec.name}: extracted to {extracted}")


def build_parser() -> argparse.ArgumentParser:
    """Parse --corpus, --dest, --force, and --no-extract."""
    parser = argparse.ArgumentParser(
        description="Download and extract the raw FLORES-200 / MASSIVE corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  download_corpus.py --corpus both\n"
            "  download_corpus.py --corpus flores --no-extract\n"
            "  download_corpus.py --corpus massive --force\n"
            "\n"
            "Archives come from the official URLs, are verified against a pinned\n"
            "SHA-256, and are extracted next to this script (override with --dest).\n"
            "Re-running skips anything already downloaded and valid."
        ),
    )
    parser.add_argument(
        "--corpus",
        choices=[*SPECS, "both"],
        required=True,
        help='which corpus to fetch: "flores" (FLORES-200, ~25 MB), '
             '"massive" (MASSIVE 1.1, ~40 MB), or "both" (required)',
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory to store archives and extracted data; defaults to this "
             "script's directory (partA/corpus). Example: --dest /data/corpora",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore existing files and start clean: re-download the archive and "
             "re-extract even when both are already valid. Use to recover from a "
             "corrupt or partial copy",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="download and verify the archive only, skip extraction; rerun "
             "without this flag later to extract it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    specs = list(SPECS.values()) if args.corpus == "both" else [SPECS[args.corpus]]
    try:
        for spec in specs:
            process_corpus(spec, args.dest, force=args.force, no_extract=args.no_extract)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
