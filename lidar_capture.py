# lidar_capture.py
import time
from scapy.all import sniff, wrpcap

PACKET_FILE = "lidar_packets.pcap"
LIDAR_INTERFACE = "Ethernet" 
CAPTURE_DURATION = 10 * 60 

def handle_packet(pkt):
    """
    Guarda cada paquete tal como llega inmediatamente.
    """
    wrpcap(PACKET_FILE, [pkt], append=True)
    print(f"Paquete guardado: {len(pkt)} bytes")

def start_capture():
    print(f"=== Capturando paquetes LiDAR en {LIDAR_INTERFACE} durante 10 minutos ===")
    start_time = time.time()

    try:
        sniff(
            iface=LIDAR_INTERFACE,
            store=False,
            prn=handle_packet,
            promisc=True,
            stop_filter=lambda pkt: time.time() - start_time > CAPTURE_DURATION
        )
    except KeyboardInterrupt:
        print("\nCaptura interrumpida por teclado. Todos los paquetes ya guardados.")
    finally:
        print("Captura finalizada.")

if __name__ == "__main__":
    start_capture()
