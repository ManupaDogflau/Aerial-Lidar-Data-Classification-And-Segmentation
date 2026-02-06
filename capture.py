# capture.py
from scapy.all import sniff, Ether
import matplotlib.cm as cm

from config import *
from parser import parse_payload, is_valid_return
from geometry import polar_to_xyz

point_buffer = []
color_buffer = []
packet_count = 0

def firing_to_xyz(firing):
    from config import VERTICAL_ANGLES_DEG
    xyz = []

    for ch, ((d1,i1),(d2,i2)) in enumerate(zip(firing["echo1"], firing["echo2"])):
        az = firing["azimuth"][ch]
        vert = VERTICAL_ANGLES_DEG[ch]

        if is_valid_return(d1, i1):
            xyz.append((*polar_to_xyz(d1, az, vert), i1))
        if is_valid_return(d2, i2):
            xyz.append((*polar_to_xyz(d2, az, vert), i2))

    return xyz

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

    for firing in firings:
        for p in firing_to_xyz(firing):
            point_buffer.append(p)
            intensity = p[3]
            r, g, b, _ = cm.jet(intensity / 255.0)
            color_buffer.append((r, g, b))

    packet_count += 1
    print(f"Captured packet {packet_count}, total points: {len(point_buffer)}")

    if packet_count >= PACKETS_TO_CAPTURE:
        raise KeyboardInterrupt
