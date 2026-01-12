#!/usr/bin/env python3
"""
Simple EarthScope FDSN (dataselect) query with ObsPy -> write MiniSEED.

Dependencies:
  pip install obspy

Notes:
  EarthScope FDSN web services are available at https://service.earthscope.org/fdsnws/
  including dataselect v1 for MiniSEED waveforms.
"""

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

def main():
    # EarthScope FDSN base URL (station + dataselect + more)
    client = Client(base_url="https://service.earthscope.org/") 
    # client = Client("IRIS")

    # --- Edit these to your target data ---
    network = "IU"
    station = "ANMO"
    location = "00"
    channel = "BHZ"
    starttime = UTCDateTime("2014-05-01T00:00:00")
    endtime   = UTCDateTime("2014-05-01T00:10:00")
    out_file = "output.mseed"
    # --------------------------------------

    try:
        st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            attach_response=False,
        )
    except FDSNNoDataException:
        print("No data returned for that request.")
        return

    if len(st) == 0:
        print("Empty stream returned.")
        return

    st.write(out_file, format="MSEED")
    print(f"Wrote {out_file} with {len(st)} trace(s).")

if __name__ == "__main__":
    main()
