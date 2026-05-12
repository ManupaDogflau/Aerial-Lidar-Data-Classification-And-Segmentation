# main.py
import threading
from scapy.all import sniff

from capture import handle_packet, set_streamer
from streamer import LivePointStreamer
from pointcloud_saver import PointCloudSaver
from config import LIDAR_IP

print("=== Live LiDAR Streaming ===")
print(f"Listening for LiDAR at {LIDAR_IP}")

streamer = LivePointStreamer()
set_streamer(streamer)

saver = PointCloudSaver(streamer, save_interval_ms=1000)

sniff_thread = threading.Thread(
    target=lambda: sniff(
        iface="Ethernet",
        store=False,
        prn=handle_packet,
        promisc=True,
        stop_filter=lambda pkt: not streamer.running
    ),
    daemon=True
)

sniff_thread.start()

saver.start()

try:
    streamer.start()
except KeyboardInterrupt:
    print("\n[INFO] Deteniendo...")
finally:
    saver.stop()
    streamer.stop()