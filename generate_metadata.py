"""
generate_metadata.py
====================
Run this script locally (not in WASM) whenever the OSSL dataset list changes.
It reads the same CSV that the Shiny app uses, fetches only the Parquet *schema*
(footer bytes only — no full download) for every file, and writes a compact JSON
that the Shiny app can fetch at startup instead of downloading any Parquet.

Output
------
  ossl_metadata.json   (place this next to app.py, or host it on your bucket)

Usage
-----
  pip install pandas pyarrow requests
  python generate_metadata.py

  # Optionally point to a different CSV:
  python generate_metadata.py --csv path/to/ossl_individual_datasets_urls_v1.3.csv

  # Write output elsewhere:
  python generate_metadata.py --out path/to/ossl_metadata.json

Schema of the produced JSON
----------------------------
{
  "generated_at": "2026-05-15T12:00:00Z",   // ISO-8601 UTC timestamp
  "datasets": {
    "<dataset_code>": {
      "spectra_types": ["mir", "visnir"],    // available spectral types
      "files": {
        "soilsite": {
          "url": "https://...",
          "columns": ["col_a", "col_b", ...]
        },
        "soillab": {
          "url": "https://...",
          "columns": ["col_a", "col_b", ...]
        },
        "mir": {                             // one key per spectra type
          "url": "https://...",
          "scan_min": 600.0,                 // min numeric wavelength/wavenumber
          "scan_max": 4000.0,                // max numeric wavelength/wavenumber
          "meta_columns": ["col_a", ...],    // non-scan metadata columns
          "scan_count": 1700                 // number of scan_ columns
        },
        "visnir": { ... }
      }
    },
    ...
  }
}
"""

import argparse
import io
import json
import re
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Disable SSL verification for environments with cert issues (same as app.py)
ssl._create_default_https_context = ssl._create_unverified_context

SCAN_PREFIX = "scan_"
SKIP_COLS = {"id.layer_local_c", "id.scan_local_c", "id.layer_uuid_c", "id.layer_uuid"}


def fetch_schema(url: str) -> pq.ParquetSchema:
    """
    Download only the Parquet footer (schema) — not the full file.

    Parquet stores its footer at the end of the file.  The footer size is
    encoded in the last 8 bytes: bytes [-8:-4] are the footer length as a
    little-endian int32, bytes [-4:] are the magic bytes b'PAR1'.

    We use two ranged HTTP requests:
      1. Fetch the last 8 bytes to learn the footer length (F).
      2. Fetch the last (F + 8) bytes, reconstruct a minimal valid Parquet
         buffer, and hand it to pyarrow.

    This keeps the download to a few KB regardless of dataset size.
    Falls back to full download if the server does not support Range requests.
    """
    MAGIC = b"PAR1"

    req_tail = urllib.request.Request(url, headers={"Range": "bytes=-8"})
    try:
        with urllib.request.urlopen(req_tail) as resp:
            tail = resp.read()
            supports_range = resp.status == 206
    except Exception:
        supports_range = False
        tail = b""

    if supports_range and len(tail) == 8 and tail[-4:] == MAGIC:
        footer_len = int.from_bytes(tail[:4], "little")
        fetch_len = footer_len + 8  # footer + its length field + magic

        req_footer = urllib.request.Request(
            url, headers={"Range": f"bytes=-{fetch_len}"}
        )
        with urllib.request.urlopen(req_footer) as resp:
            footer_bytes = resp.read()

        # Reconstruct minimal valid Parquet: magic + footer + len + magic
        parquet_buf = MAGIC + footer_bytes
        buf = io.BytesIO(parquet_buf)
        try:
            return pq.read_schema(buf)
        except Exception:
            pass  # fall through to full download

    # Fallback: full download (server doesn't support Range or parse failed)
    print(f"    [!] Range request failed for {url!r}, falling back to full download")
    with urllib.request.urlopen(url) as resp:
        buf = io.BytesIO(resp.read())
    return pq.read_schema(buf)


def parse_scan_cols(schema: pq.ParquetSchema):
    """
    Given a Parquet schema, return:
      - scan_values : sorted list of float wavelength/wavenumber values
      - meta_cols   : non-scan, non-id column names
    """
    all_names = schema.names
    scan_values = []
    meta_cols = []

    for c in all_names:
        if str(c).startswith(SCAN_PREFIX):
            num_str = re.sub(r"^scan_.*?\.", "", str(c))
            num_str = re.sub(r"_(abs|ref|bc\.abs)$", "", num_str)
            try:
                scan_values.append(float(num_str))
            except ValueError:
                pass
        elif c not in SKIP_COLS:
            meta_cols.append(c)

    scan_values.sort()
    return scan_values, sorted(meta_cols)


def spectra_type_from_filename(filename: str) -> str | None:
    """Return 'mir', 'visnir', or 'nir' based on the filename, or None."""
    fname = filename.lower()
    if "_mir_" in fname:
        return "mir"
    if "_visnir_" in fname:
        return "visnir"
    if "_nir_" in fname:
        return "nir"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_metadata(csv_path: Path) -> dict:
    urls_df = pd.read_csv(csv_path)
    # Keep only parquet rows
    urls_df = urls_df[urls_df["ossl_file"].str.contains(r"\.parquet", na=False)]

    dataset_codes = sorted(urls_df["dataset_code"].unique().tolist())
    total = len(dataset_codes)

    datasets: dict = {}

    for idx, code in enumerate(dataset_codes, 1):
        print(f"\n[{idx}/{total}] {code}")
        subset = urls_df[urls_df["dataset_code"] == code]
        entry: dict = {"spectra_types": [], "files": {}}

        # --- soilsite ---
        site_rows = subset[subset["ossl_file"].str.contains("soilsite")]
        if not site_rows.empty:
            url = site_rows["public_url"].values[0]
            print(f"  soilsite  -> {url.split('/')[-1]}")
            try:
                schema = fetch_schema(url)
                entry["files"]["soilsite"] = {
                    "url": url,
                    "columns": sorted(schema.names),
                }
            except Exception as exc:
                print(f"    ERROR: {exc}")

        # --- soillab ---
        lab_rows = subset[subset["ossl_file"].str.contains("soillab")]
        if not lab_rows.empty:
            url = lab_rows["public_url"].values[0]
            print(f"  soillab   -> {url.split('/')[-1]}")
            try:
                schema = fetch_schema(url)
                entry["files"]["soillab"] = {
                    "url": url,
                    "columns": sorted(schema.names),
                }
            except Exception as exc:
                print(f"    ERROR: {exc}")

        # --- spectra (mir / visnir / nir) ---
        spec_rows = subset[
            ~subset["ossl_file"].str.contains("soilsite")
            & ~subset["ossl_file"].str.contains("soillab")
        ]
        for _, row in spec_rows.iterrows():
            stype = spectra_type_from_filename(row["ossl_file"])
            if stype is None:
                continue
            url = row["public_url"]
            print(f"  {stype:<8}  -> {url.split('/')[-1]}")
            try:
                schema = fetch_schema(url)
                scan_values, meta_cols = parse_scan_cols(schema)
                entry["files"][stype] = {
                    "url": url,
                    "scan_min": scan_values[0] if scan_values else None,
                    "scan_max": scan_values[-1] if scan_values else None,
                    "scan_count": len(scan_values),
                    "meta_columns": meta_cols,
                }
                if stype not in entry["spectra_types"]:
                    entry["spectra_types"].append(stype)
            except Exception as exc:
                print(f"    ERROR: {exc}")

        entry["spectra_types"].sort()
        datasets[code] = entry

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": datasets,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ossl_metadata.json")
    parser.add_argument(
        "--csv",
        default="input/ossl_individual_datasets_urls_v1.3.csv",
        help="Path to the dataset URLs CSV (default: input/ossl_individual_datasets_urls_v1.3.csv)",
    )
    parser.add_argument(
        "--out",
        default="ossl_metadata.json",
        help="Output JSON path (default: ossl_metadata.json)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading URLs from: {csv_path}")
    metadata = build_metadata(csv_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    n = len(metadata["datasets"])
    print(f"\nDone. Wrote {n} datasets to {out_path}")
    print(f"Generated at: {metadata['generated_at']}")


if __name__ == "__main__":
    main()