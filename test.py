from obspy.clients.fdsn import Client
from obspy import UTCDateTime


# https://service.earthscope.org/fdsnws/dataselect/1/query?net=IU&sta=ANMO&loc=00&cha=BHZ&start=2010-02-27T06:30:00.000&end=2010-02-27T10:30:00.000

# Initialize client and define parameters
client = Client("IRIS")

starttime = UTCDateTime("2010-02-27T06:30:00.000")
endtime = UTCDateTime("2010-02-27T10:30:00.000")

# Request waveform data
try:
    st = client.get_waveforms(network="IU", 
                              station="ANMO", 
                              location="00", 
                              channel="BHZ", 
                              starttime=starttime, 
                              endtime=endtime)
    
    print("\nStream information from FDSN client request:")
    print(st)
except Exception as e:
    print(f"Error during FDSN request: {e}")

