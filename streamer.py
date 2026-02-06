# streamer.py
import open3d as o3d
import numpy as np
import threading
import time

MAX_POINTS = 30_000

class LivePointStreamer:
    def __init__(self):
        self.points = []
        self.colors = []
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
        self.vis.create_window(
            window_name="Live LiDAR Stream",
            width=1280,
            height=720
        )

        self.vis.add_geometry(self.axes)

        while self.running:
            with self.lock:
                if self.points:
                    self.pcd.points = o3d.utility.Vector3dVector(
                        np.asarray(self.points)
                    )
                    self.pcd.colors = o3d.utility.Vector3dVector(
                        np.asarray(self.colors)
                    )

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
        with self.lock:
            self.points.extend(new_points)
            self.colors.extend(new_colors)

            if len(self.points) > MAX_POINTS:
                self.points = self.points[-MAX_POINTS:]
                self.colors = self.colors[-MAX_POINTS:]


    def stop(self):
        self.running = False
