# streamer.py
import open3d as o3d
import numpy as np
import threading
import time

POINT_TTL = 1


class LivePointStreamer:
    def __init__(self,visualize=False):
        self.visualize = visualize
        self.points = np.empty((0, 3), dtype=np.float64)
        self.colors = np.empty((0, 3), dtype=np.float64)
        self.timestamps = np.empty((0,), dtype=np.float64)

        self.lock = threading.Lock()
        self.running = True

        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.geometry_added = False

        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5,
            origin=[0, 0, 0]
        )

    def start(self):

        if not self.visualize:
            while self.running:
                self.cleanup_old_points()
                time.sleep(0.03)
            return

        self.vis.create_window(
            window_name="Live LiDAR Stream",
            width=1280,
            height=720
        )

        self.vis.add_geometry(self.axes)

        while self.running:
            with self.lock:
                self.cleanup_old_points()

                if self.points.shape[0] > 0:
                    self.pcd.points = o3d.utility.Vector3dVector(self.points)
                    self.pcd.colors = o3d.utility.Vector3dVector(self.colors)

                    if not self.geometry_added:
                        self.vis.add_geometry(self.pcd)
                        ctr = self.vis.get_view_control()

                        ctr.set_lookat([0, 0, 0])
                        ctr.set_front([0, 0, 1])
                        ctr.set_up([0, 1, 0])
                        ctr.set_zoom(0.7)

                        self.geometry_added = True

            if self.geometry_added:
                self.vis.update_geometry(self.pcd)

            if not self.vis.poll_events():
                self.running = False
                break

            self.vis.update_renderer()
            time.sleep(0.03)

        self.vis.destroy_window()

    def add_points(self, new_points, new_colors):
        if len(new_points) == 0:
            return  # 🔥 evita problemas directamente

        now = time.time()

        new_points = np.asarray(new_points, dtype=np.float64).reshape(-1, 3)
        new_colors = np.asarray(new_colors, dtype=np.float64).reshape(-1, 3)

        timestamps = np.full((new_points.shape[0],), now, dtype=np.float64)

        with self.lock:
            if self.points.size == 0:
                # 🔥 caso inicial (evita vstack innecesario)
                self.points = new_points
                self.colors = new_colors
                self.timestamps = timestamps
            else:
                self.points = np.vstack((self.points, new_points))
                self.colors = np.vstack((self.colors, new_colors))
                self.timestamps = np.concatenate((self.timestamps, timestamps))
                
    def get_points(self):
        with self.lock:
            return np.copy(self.points).astype(np.float32)
        
    def cleanup_old_points(self):
        with self.lock:

            if self.points.shape[0] == 0:
                return

            now = time.time()

            mask = (now - self.timestamps) <= POINT_TTL

            self.points = self.points[mask]
            self.colors = self.colors[mask]
            self.timestamps = self.timestamps[mask]

    def stop(self):
        self.running = False
        
    def get_pointcloud(self):
        with self.lock:
            return (
                np.copy(self.points),
                np.copy(self.colors)
            )