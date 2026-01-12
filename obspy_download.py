#!/usr/bin/env python3
"""
Stream MiniSEED for stations listed in a CSV, and write ONE metadata row PER DAY.
For each day, it parses the MiniSEED and computes STATS (from data) for ONLY the first trace.

Input CSV example:
Station,DataCenter,Start,End,Site,Latitude,Longitude,Elevation
KRES,IRISDMC,2014-05-01,2016-12-31,KRES,47.758739,-122.29097,52
MA05,IRISDMC,2014-05-01,2016-12-31,MA05,46.754669,-122.226189,488

What it does (no parallel):
  - For each station row
  - For each day in the requested time range:
      1) Discover available channels via station service (level="channel")
      2) Stream waveforms via dataselect /query using requests (no MiniSEED saved)
      3) Parse MiniSEED bytes into an ObsPy Stream
      4) Take ONLY the first trace (st[0])
      5) Compute data stats for that trace (min/max/mean/std/rms, gaps count, etc.)
      6) Write one CSV row for that day

Usage:
  python daily_first_trace_stats_csv.py stations.csv output_dir [--start ISO] [--end ISO] [--max-bytes N]

Output:
  <output_dir>/daily_first_trace_stats.csv

Dependencies:
  pip install obspy requests numpy
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
from obspy import UTCDateTime, read as obspy_read
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException


# -----------------------------
# Defaults / knobs
# -----------------------------
CHANNEL = os.getenv("CHANNEL", "BH?,HH?")
LOCATION = os.getenv("LOCATION", "*")
TIMEOUT_S = float(os.getenv("TIMEOUT_S", "60"))
DEFAULT_MAX_BYTES = int(os.getenv("MAX_BYTES", str(200 * 1024 * 1024)))  # 200 MB

DATACENTER_MAP: Dict[str, str] = {
    "IRISDMC": "IRIS",
    "IRIS": "IRIS",
    "SCEDC": "SCEDC",
    "NCEDC": "NCEDC",
}


@dataclass(frozen=True)
class StationRow:
    station: str
    datacenter: str
    start: UTCDateTime
    end: UTCDateTime
    site: str
    lat: float
    lon: float
    elev_m: float


def parse_date_ymd(s: str) -> UTCDateTime:
    return UTCDateTime(s.strip())


def parse_iso_time(s: str) -> UTCDateTime:
    return UTCDateTime(s.strip())


def iter_days(t1: UTCDateTime, t2: UTCDateTime) -> Iterable[Tuple[UTCDateTime, UTCDateTime]]:
    """
    Yield [day_start, day_end) windows aligned to midnight UTC, intersected with [t1,t2).
    """
    cur = UTCDateTime(t1.date)  # midnight
    while cur < t2:
        nxt = cur + 86400
        win_start = max(cur, t1)
        win_end = min(nxt, t2)
        if win_end > win_start:
            yield (win_start, win_end)
        cur = nxt


def provider_for_datacenter(dc: str) -> str:
    key = dc.strip().upper()
    if key not in DATACENTER_MAP:
        raise ValueError(f"Unknown DataCenter '{dc}'. Add it to DATACENTER_MAP.")
    return DATACENTER_MAP[key]


def read_station_csv(csv_path: Path) -> List[StationRow]:
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
                    datacenter=r["DataCenter"].strip(),
                    start=start,
                    end=end,
                    site=(r.get("Site") or "").strip(),
                    lat=float(r["Latitude"]),
                    lon=float(r["Longitude"]),
                    elev_m=float(r["Elevation"]),
                )
            )
    return rows


def pick_station_channels(inv) -> List[tuple[str, str, str, str]]:
    out = set()
    for net in inv:
        for sta in net:
            for cha in sta:
                loc = cha.location_code or ""
                out.add((net.code, sta.code, loc, cha.code))
    return sorted(out)


def build_bulk_lines(chan_tuples: List[tuple[str, str, str, str]], t1: UTCDateTime, t2: UTCDateTime) -> str:
    lines = []
    for net, sta, loc, cha in chan_tuples:
        loc_out = loc if loc != "" else "--"
        lines.append(f"{net} {sta} {loc_out} {cha} {t1.isoformat()} {t2.isoformat()}")
    return "\n".join(lines) + "\n"


def stream_dataselect_bulk(dataselect_base_url: str, bulk_text: str, *, timeout_s: float, max_bytes: int) -> bytes:
    url = dataselect_base_url.rstrip("/") + "/query"
    headers = {"Content-Type": "text/plain", "Accept": "application/vnd.fdsn.mseed"}

    with requests.post(url, data=bulk_text.encode("utf-8"), headers=headers, stream=True, timeout=timeout_s) as r:
        if r.status_code in (204, 404):
            return b""
        r.raise_for_status()

        buf = bytearray()
        for chunk in r.iter_content(chunk_size=1024 * 128):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > max_bytes:
                raise MemoryError(
                    f"MiniSEED exceeds max_bytes={max_bytes}. Reduce date span per request or increase --max-bytes."
                )
        return bytes(buf)


def compute_trace_data_stats(tr) -> dict:
    """
    Compute stats from the actual data array for a Trace.
    Handles masked arrays, NaNs, and non-float integer types.
    """
    data = tr.data
    # Convert masked->filled and to float64 for stable stats
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)
    x = np.asarray(data, dtype=np.float64)

    # If all NaN (can happen after fill), return blanks
    finite = np.isfinite(x)
    if not np.any(finite):
        return {
            "data_min": "",
            "data_max": "",
            "data_mean": "",
            "data_std": "",
            "data_rms": "",
            "data_p2p": "",
            "finite_frac": 0.0,
        }

    xf = x[finite]
    mn = float(np.min(xf))
    mx = float(np.max(xf))
    mean = float(np.mean(xf))
    std = float(np.std(xf))
    rms = float(np.sqrt(np.mean(xf * xf)))
    p2p = float(mx - mn)
    finite_frac = float(np.sum(finite) / x.size)

    return {
        "data_min": mn,
        "data_max": mx,
        "data_mean": mean,
        "data_std": std,
        "data_rms": rms,
        "data_p2p": p2p,
        "finite_frac": finite_frac,
    }


def first_trace_daily_row(
    station_row: StationRow,
    provider: str,
    day_start: UTCDateTime,
    day_end: UTCDateTime,
    tr,
) -> dict:
    stats = tr.stats
    out = {
        # CSV context
        "csv_station": station_row.station,
        "csv_datacenter": station_row.datacenter,
        "provider": provider,
        "csv_site": station_row.site,
        "csv_latitude": station_row.lat,
        "csv_longitude": station_row.lon,
        "csv_elevation_m": station_row.elev_m,
        # day + request window
        "day": day_start.date.strftime("%Y-%m-%d"),
        "request_start": day_start.isoformat(),
        "request_end": day_end.isoformat(),
        # first-trace header fields
        "trace_id": tr.id,
        "network": getattr(stats, "network", ""),
        "station": getattr(stats, "station", ""),
        "location": getattr(stats, "location", ""),
        "channel": getattr(stats, "channel", ""),
        "sampling_rate": getattr(stats, "sampling_rate", ""),
        "npts": getattr(stats, "npts", ""),
        "starttime": stats.starttime.isoformat() if getattr(stats, "starttime", None) else "",
        "endtime": stats.endtime.isoformat() if getattr(stats, "endtime", None) else "",
        "delta": getattr(stats, "delta", ""),
        "calib": getattr(stats, "calib", ""),
    }
    out.update(compute_trace_data_stats(tr))
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Daily streaming MiniSEED -> first trace -> compute data stats -> write CSV (no parallel)."
    )
    p.add_argument("csv_path", help="Input station CSV")
    p.add_argument("output_dir", help="Directory to write output CSV")
    p.add_argument("--start", default=None, help="Override start time for ALL stations (ISO-8601 UTC).")
    p.add_argument("--end", default=None, help="Override end time for ALL stations (ISO-8601 UTC).")
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Max bytes to buffer per day request.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    override_start = parse_iso_time(args.start) if args.start else None
    override_end = parse_iso_time(args.end) if args.end else None

    rows = read_station_csv(csv_path)
    if not rows:
        print("No stations found in CSV.", file=sys.stderr)
        raise SystemExit(1)

    out_csv = out_dir / "daily_first_trace_stats.csv"

    fieldnames = [
        "csv_station",
        "csv_datacenter",
        "provider",
        "csv_site",
        "csv_latitude",
        "csv_longitude",
        "csv_elevation_m",
        "day",
        "request_start",
        "request_end",
        "trace_id",
        "network",
        "station",
        "location",
        "channel",
        "sampling_rate",
        "npts",
        "starttime",
        "endtime",
        "delta",
        "calib",
        # computed from data (first trace)
        "data_min",
        "data_max",
        "data_mean",
        "data_std",
        "data_rms",
        "data_p2p",
        "finite_frac",
    ]

    wrote_header = False
    total_rows_written = 0

    print(f"Stations: {len(rows)} | CHANNEL={CHANNEL} | LOCATION={LOCATION} | max_bytes={args.max_bytes}")
    if override_start or override_end:
        print(f"Override window: start={override_start} end={override_end}")
    print(f"Writing CSV: {out_csv}")

    for i, row in enumerate(rows, start=1):
        provider = provider_for_datacenter(row.datacenter)
        client = Client(provider, timeout=TIMEOUT_S)
        dataselect_base = client.base_url.rstrip("/") + "/fdsnws/dataselect/1"

        start = override_start if override_start is not None else row.start
        end = override_end if override_end is not None else row.end
        if end <= start:
            print(f"ERR  [{i}] {row.station}: end<=start")
            continue

        print(f"\n[{i}/{len(rows)}] {row.station} ({row.datacenter}->{provider}) {start.isoformat()} -> {end.isoformat()}")

        for day_start, day_end in iter_days(start, end):
            # 1) station discovery for this day
            try:
                inv = client.get_stations(
                    network="*",
                    station=row.station,
                    location=LOCATION,
                    channel=CHANNEL,
                    starttime=day_start,
                    endtime=day_end,
                    level="channel",
                    includerestricted=False,
                )
            except FDSNNoDataException:
                continue
            except Exception as e:
                print(f"  {day_start.date}: ERR stations: {e}")
                continue

            chan_tuples = pick_station_channels(inv)
            if not chan_tuples:
                continue

            # 2) stream mseed bytes
            bulk_text = build_bulk_lines(chan_tuples, day_start, day_end)
            try:
                mseed_bytes = stream_dataselect_bulk(dataselect_base, bulk_text, timeout_s=TIMEOUT_S, max_bytes=args.max_bytes)
            except MemoryError as e:
                print(f"  {day_start.date}: ERR dataselect: {e}")
                continue
            except requests.HTTPError as e:
                print(f"  {day_start.date}: ERR dataselect http: {e}")
                continue
            except Exception as e:
                print(f"  {day_start.date}: ERR dataselect: {e}")
                continue

            if not mseed_bytes:
                continue

            # 3) parse and take first trace
            try:
                st = obspy_read(io.BytesIO(mseed_bytes))
            except Exception as e:
                print(f"  {day_start.date}: ERR parse mseed: {e}")
                continue
            if len(st) == 0:
                continue

            tr0 = st[0]

            # 4) compute stats from first trace data + write one row
            meta = first_trace_daily_row(row, provider, day_start, day_end, tr0)

            with out_csv.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not wrote_header:
                    writer.writeheader()
                    wrote_header = True
                writer.writerow(meta)

            total_rows_written += 1
            print(f"  {day_start.date}: wrote stats for {meta['trace_id']} (npts={meta['npts']})")

    print(f"\nDone. Total daily rows written: {total_rows_written}")


if __name__ == "__main__":
    main()
