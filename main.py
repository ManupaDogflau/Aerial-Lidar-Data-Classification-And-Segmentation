# main.py
from scapy.all import sniff
from capture import handle_packet, point_buffer, color_buffer
from pcd_writer import write_pcd
from visualization import visualize
from config import LIDAR_IP

print("=== LS-LiDAR Dual-Echo 16-Line Capture ===")
print(f"Listening for LiDAR at {LIDAR_IP}")

try:
    sniff(
        iface="Ethernet",
        store=False,
        prn=handle_packet,
        promisc=True
    )
except KeyboardInterrupt:
    pass

pcd_file = "lidar_cloud.pcd"
write_pcd(pcd_file, point_buffer, colors=color_buffer)
print(f"✅ Saved {len(point_buffer)} points to {pcd_file}")

visualize(pcd_file)
