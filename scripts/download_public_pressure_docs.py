"""Download public NASA pressure-system PDFs for parser stress tests.

The PDFs are public NASA standards, not licensed ASME material. They are kept
separate from the SPES-1 correctness benchmark and used only for parser,
retrieval, and citation stress testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "parser_benchmark" / "public_pressure_sources.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "parser_benchmark" / "pdfs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download public NASA pressure-system PDFs."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="JSON manifest of public pressure-system documents.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PDFs should be written.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when they already exist.",
    )
    return p.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _download(url: str, dest: Path) -> None:
    request = Request(url, headers={"User-Agent": "ANVIL parser benchmark"})
    try:
        with urlopen(request, timeout=120) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    if not data.startswith(b"%PDF"):
        raise RuntimeError(
            f"Downloaded payload for {url} is not a PDF "
            f"(Content-Type={content_type!r})."
        )
    dest.write_bytes(data)


def main() -> None:
    args = _parse_args()
    manifest_path = _resolve(args.manifest)
    output_dir = _resolve(args.output_dir)
    sources = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        filename = source["filename"]
        url = source["url"]
        dest = output_dir / filename
        if dest.exists() and not args.force:
            print(f"exists {dest}")
            continue
        _download(url, dest)
        print(f"downloaded {dest}")


if __name__ == "__main__":
    main()
