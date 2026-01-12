#!/usr/bin/env python3
"""
Row-by-row MiniSEED download from an IRIS/EarthScope FDSN endpoint and compute mean per trace.

Input CSV format:
Station,DataCenter,Start,End,Site,Latitude,Longitude,Elevation
MA05,IRISDMC,2014-05-01,2016-12-31,MA05,46.754669,-122.226189,488

Behavior:
  - Sequential loop over CSV rows (no parallel)
  - Start/End are date-only (YYYY-MM-DD), treated as midnight UTC
  - End date is inclusive (+86400 seconds)
  - Downloads waveforms using ObsPy FDSN client
  - Writes one MiniSEED file per CSV row
  - Computes mean for each trace and writes trace_means.csv

Defaults (override with env vars):
  NETWORK="*"
  LOCATION="*"
  CHANNEL="BH?,HH?"

Usage:
  python station_csv_download_means_earthscope.py stations.csv output_dir
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException


# -----------------------------
# Config
# -----------------------------
# Use the active IRIS/EarthScope FDSN base URL (more reliable than Client("IRIS") shortcut).
FDSN_BASE_URL = os.getenv("FDSN_BASE_URL", "https://service.earthscope.org/fdsnws/dataselect/1")

NETWORK = os.getenv("NETWORK", "*")
LOCATION = os.getenv("LOCATION", "*")
CHANNEL = os.getenv("CHANNEL", "BH?,HH?")

TIMEOUT_S = float(os.getenv("TIMEOUT_S", "120"))  # longer helps service discovery
ATTACH_RESPONSE = os.getenv("ATTACH_RESPONSE", "0") == "1"


@dataclass(frozen=True)
class StationRow:
    station: str
    start: UTCDateTime
    end: UTCDateTime
    site: str
    lat: float
    lon: float
    elev_m: float


def parse_date_ymd(s: str) -> UTCDateTime:
    """Parse YYYY-MM-DD into UTCDateTime at midnight UTC."""
    return UTCDateTime(s.strip())


def safe_name(s: str) -> str:
    """Make timestamps filesystem-safe."""
    return s.replace(":", "").replace("/", "-").replace(" ", "")


def trace_mean(tr) -> float:
    """Compute mean of trace samples (float64, NaN-safe)."""
    data = tr.data
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)
    x = np.asarray(data, dtype=np.float64)
    return float(np.nanmean(x))


def read_station_csv(csv_path: Path) -> List[StationRow]:
    """
    Read station CSV into StationRow objects.
    DataCenter column is accepted but ignored.
    """
    rows: List[StationRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"Station", "DataCenter", "Start", "End", "Site", "Latitude", "Longitude", "Elevation"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for r in reader:
            if not any((v or "").strip() for v in r.values()):
                continue

            start = parse_date_ymd(r["Start"])
            end = parse_date_ymd(r["End"]) + 86400  # inclusive end date

            rows.append(
                StationRow(
                    station=r["Station"].strip(),
                    start=start,
                    end=end,
                    site=(r.get("Site") or "").strip(),
                    lat=float(r["Latitude"]),
                    lon=float(r["Longitude"]),
                    elev_m=float(r["Elevation"]),
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Download MiniSEED per CSV row and compute mean per trace.")
    ap.add_argument("csv_path", help="CSV: Station,DataCenter,Start,End,Site,Latitude,Longitude,Elevation")
    ap.add_argument("output_dir", help="Directory to write MiniSEED files and trace_means.csv")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, help="Network timeout (seconds)")
    args = ap.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_station_csv(csv_path)
    if not rows:
        print("No rows found in CSV.")
        return

    # Create the client once (single provider/base URL)
    client = Client(base_url="https://service.earthscope.org/fdsnws/dataselect/1", timeout=args.timeout)

    means_csv = out_dir / "trace_means.csv"
    fieldnames = [
        "row_index",
        "station",
        "base_url",
        "network",
        "location",
        "channel",
        "request_start",
        "request_end",
        "mseed_file",
        "trace_id",
        "trace_starttime",
        "trace_endtime",
        "sampling_rate",
        "npts",
        "mean",
    ]
    wrote_header = False

    print(
        f"Rows={len(rows)} base_url={FDSN_BASE_URL} "
        f"NETWORK={NETWORK} LOCATION={LOCATION} CHANNEL={CHANNEL} timeout={args.timeout}"
    )

    for i, row in enumerate(rows, start=1):
        t1, t2 = row.start, row.end
        print(f"[{i}] {row.station} {t1.isoformat()} -> {t2.isoformat()}")

        try:
            st = client.get_waveforms(
                network=NETWORK,
                station=row.station,
                location=LOCATION,
                channel=CHANNEL,
                starttime=t1,
                endtime=t2,
                attach_response=ATTACH_RESPONSE,
            )
        except FDSNNoDataException:
            print("  no data")
            continue
        except Exception as e:
            print(f"  ERROR: download failed: {e}")
            continue

        if len(st) == 0:
            print("  empty stream")
            continue

        station_dir = out_dir / row.station
        station_dir.mkdir(parents=True, exist_ok=True)

        mseed_name = f"{row.station}_{safe_name(t1.isoformat())}_to_{safe_name(t2.isoformat())}.mseed"
        mseed_path = station_dir / mseed_name

        try:
            st.write(str(mseed_path), format="MSEED")
        except Exception as e:
            print(f"  ERROR: could not write MiniSEED: {e}")
            continue

        print(f"  wrote {mseed_path} (traces={len(st)})")

        # Append per-trace means
        with means_csv.open("a", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            if not wrote_header:
                writer.writeheader()
                wrote_header = True

            for tr in st:
                stats = tr.stats
                writer.writerow(
                    {
                        "row_index": i,
                        "station": row.station,
                        "base_url": FDSN_BASE_URL,
                        "network": NETWORK,
                        "location": LOCATION,
                        "channel": CHANNEL,
                        "request_start": t1.isoformat(),
                        "request_end": t2.isoformat(),
                        "mseed_file": str(mseed_path),
                        "trace_id": tr.id,
                        "trace_starttime": stats.starttime.isoformat() if stats.starttime else "",
                        "trace_endtime": stats.endtime.isoformat() if stats.endtime else "",
                        "sampling_rate": getattr(stats, "sampling_rate", ""),
                        "npts": getattr(stats, "npts", ""),
                        "mean": trace_mean(tr),
                    }
                )

    print(f"\nDone.")
    print(f"MiniSEED written under: {out_dir}")
    print(f"Trace means CSV: {means_csv}")


if __name__ == "__main__":
    main()
