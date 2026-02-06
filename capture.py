# capture.py
from scapy.all import sniff, Ether
import matplotlib.cm as cm

from config import *
from parser import parse_payload, is_valid_return
from geometry import polar_to_xyz

streamer = None
packet_count = 0

def set_streamer(s):
    global streamer
    streamer = s

def firing_to_xyz(firing):
    from config import VERTICAL_ANGLES_DEG
    pts = []
    cols = []

    for ch, ((d1,i1),(d2,i2)) in enumerate(zip(firing["echo1"], firing["echo2"])):
        az = firing["azimuth"][ch]
        vert = VERTICAL_ANGLES_DEG[ch]

        for d, i in ((d1,i1), (d2,i2)):
            if is_valid_return(d, i):
                x, y, z = polar_to_xyz(d, az, vert)
                pts.append((x, y, z))
                norm_i = min(1.0, i / 100.0)
                r, g, b, _ = cm.jet(norm_i)
                cols.append((r, g, b))

    return pts, cols

def handle_packet(pkt):
    global packet_count

    if not pkt.haslayer(Ether):
        return

    raw = bytes(pkt)
    if len(raw) != TOTAL_FRAME_LEN:
        return

    if pkt.haslayer("IP") and pkt["IP"].src != LIDAR_IP:
        return

    payload = raw[ETH_HEADER_LEN:ETH_HEADER_LEN + PAYLOAD_LEN]
    firings = parse_payload(payload)

    all_pts = []
    all_cols = []

    for firing in firings:
        pts, cols = firing_to_xyz(firing)
        all_pts.extend(pts)
        all_cols.extend(cols)

    if streamer:
        streamer.add_points(all_pts, all_cols)

    packet_count += 1
    print(f"Packets: {packet_count}, points: {len(all_pts)}")
