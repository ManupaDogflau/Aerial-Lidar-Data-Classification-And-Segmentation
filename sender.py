# sender.py

import socket
import threading
import time
import numpy as np
import struct


class PointCloudSender:

    def __init__(
        self,
        streamer,
        receiver_ip,
        receiver_port=5005,
        interval=0.1,
        reconnect_interval=2.0
    ):

        self.streamer = streamer

        self.receiver_ip = receiver_ip
        self.receiver_port = receiver_port

        self.interval = interval
        self.reconnect_interval = reconnect_interval

        self.sock = None
        self.running = True

    def start(self):

        threading.Thread(
            target=self._run,
            daemon=True
        ).start()

    def _connect(self):

        while self.running:

            try:

                self.sock = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM
                )

                self.sock.connect(
                    (
                        self.receiver_ip,
                        self.receiver_port
                    )
                )

                print(
                    f"[SENDER] Conectado a "
                    f"{self.receiver_ip}:{self.receiver_port}"
                )

                return

            except OSError:

                print(
                    "[SENDER] Esperando receptor..."
                )

                try:
                    self.sock.close()
                except Exception:
                    pass

                time.sleep(
                    self.reconnect_interval
                )

    def _run(self):

        self._connect()

        while self.running:

            time.sleep(self.interval)

            points = self.streamer.get_points()

            if points.size == 0:
                continue

            header = struct.pack(
                "<I",
                len(points)
            )

            payload = points.astype(
                np.float32
            ).tobytes()

            try:

                self.sock.sendall(header)
                self.sock.sendall(payload)

            except OSError:

                print(
                    "[SENDER] Conexión perdida."
                )

                try:
                    self.sock.close()
                except Exception:
                    pass

                self._connect()

    def stop(self):

        self.running = False

        if self.sock is not None:

            try:
                self.sock.shutdown(
                    socket.SHUT_RDWR
                )
            except OSError:
                pass

            try:
                self.sock.close()
            except OSError:
                pass