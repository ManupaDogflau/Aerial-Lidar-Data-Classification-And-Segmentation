# visualization.py
import open3d as o3d

def visualize(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([pcd, axes])
