"""Download five PUBLIC monthly economic series from FRED; no API key required.

Invoke manually or in an opt-in workflow. This is NOT part of default CI to
avoid converting a remote service outage into a repository-test failure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SERIES = ("BAA", "GS10", "UNRATE", "INDPRO", "FEDFUNDS")


def fetch_series(name: str, output: Path) -> Path:
    if name not in SERIES:
        raise ValueError("Invalid research-series identifier")
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    url = base + "?" + urlencode({"id": name})
    request = Request(url, headers={"User-Agent": "Research-Portfolio/1.0"})
    with urlopen(request, timeout=40) as response:
        body = response.read(20_000_000)
    beginning = body[:300].decode("utf-8-sig", errors="ignore")
    if not (("observation_date" in beginning or "DATE" in beginning)
            and name in beginning):
        raise RuntimeError(f"Unexpected non-CSV response for {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(body)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/fred"))
    a = parser.parse_args()
    for series in SERIES:
        file = fetch_series(series, a.output_dir / f"{series}.csv")
        print(f"Downloaded public FRED series {series}: {file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
