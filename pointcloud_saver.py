# pointcloud_saver.py
import os
import datetime
import open3d as o3d
import numpy as np
import threading
import time


class PointCloudSaver:
    def __init__(self, streamer, save_interval_ms=200):
        """
        streamer: instancia de LivePointStreamer
        save_interval_ms: intervalo en milisegundos
        """
        self.streamer = streamer
        self.save_interval = save_interval_ms / 1000.0  # convertir a segundos
        self.running = True

        # Crear carpeta con fecha y hora
        now = datetime.datetime.now()
        folder_name = now.strftime("scan_%Y-%m-%d_%H-%M-%S")
        self.base_path = os.path.join("captures", folder_name)

        os.makedirs(self.base_path, exist_ok=True)

        print(f"[INFO] Guardando capturas en: {self.base_path}")
        print(f"[INFO] Intervalo: {save_interval_ms} ms")

    def start(self):
        thread = threading.Thread(target=self._run)
        thread.daemon = True
        thread.start()

    def _run(self):
        counter = 0

        while self.running:
            time.sleep(self.save_interval)

            with self.streamer.lock:
                if self.streamer.points.size==0:
                    continue

                # Copia para evitar problemas de concurrencia
                points = np.copy(np.asarray(self.streamer.points))
                colors = np.copy(np.asarray(self.streamer.colors))

            # Crear nube
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Timestamp en milisegundos
            timestamp = int(time.time() * 1000)

            filename = os.path.join(
                self.base_path,
                f"cloud_{counter:04d}_{timestamp}.ply"
            )

            o3d.io.write_point_cloud(filename, pcd)

            print(f"[SAVE] {filename}")
            counter += 1

    def stop(self):
        self.running = False