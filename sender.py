# sender.py

import socket
import threading
import time
import numpy as np
import struct



class PointCloudSender:

    def __init__(self,
                 streamer,
                 receiver_ip,
                 receiver_port=5005,
                 interval=0.1):

        self.streamer = streamer
        self.interval = interval

        self.sock = socket.socket(socket.AF_INET,
                                  socket.SOCK_STREAM)

        self.receiver_ip = receiver_ip
        self.receiver_port = receiver_port

        self.running = True

    def start(self):

        self.sock.connect((self.receiver_ip,
                           self.receiver_port))

        threading.Thread(
            target=self._run,
            daemon=True
        ).start()

    def _run(self):

        while self.running:

            time.sleep(self.interval)

            points = self.streamer.get_points()

            if points.size == 0:
                continue
            
            header = struct.pack("<I", len(points))

            payload = points.astype(np.float32).tobytes()

            try:
                self.sock.sendall(header)
                self.sock.sendall(payload)
            except OSError:
                print("[SENDER] Conexión perdida.")
                self.running = False
                break
        
    def stop(self):
        self.running = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()