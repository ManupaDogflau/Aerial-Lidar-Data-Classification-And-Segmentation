# lidar_replay_thread.py
import threading
import time
from scapy.all import rdpcap

from capture import handle_packet, set_streamer
from streamer import LivePointStreamer
from pointcloud_saver import PointCloudSaver  

PACKET_FILE = "lidar_packets.pcap"
PACKET_DELAY = 0.003  # segundos entre paquetes


class PacketReplayer(threading.Thread):
    def __init__(self, pcap_file, delay=0.01):
        super().__init__(daemon=True)
        self.pcap_file = pcap_file
        self.delay = delay
        self.running = False

    def run(self):
        print(f"=== Reproduciendo paquetes desde {self.pcap_file} ===")
        packets = rdpcap(self.pcap_file)
        self.running = True

        for pkt in packets:
            if not self.running:
                break
            handle_packet(pkt)
            time.sleep(self.delay)

        print("Reproducción de paquetes finalizada.")

    def stop(self):
        self.running = False


if __name__ == "__main__":
    # Inicializa streamer
    streamer = LivePointStreamer()
    set_streamer(streamer)

    saver = PointCloudSaver(streamer, save_interval_ms=1000)

    # Inicia replayer
    replayer = PacketReplayer(PACKET_FILE, delay=PACKET_DELAY)
    replayer.start()

    saver.start()

    try:
        # Inicia visualizador
        streamer.start()
    except KeyboardInterrupt:
        print("Interrupción por teclado. Deteniendo...")
    finally:
        saver.stop()
        replayer.stop()
        streamer.stop()