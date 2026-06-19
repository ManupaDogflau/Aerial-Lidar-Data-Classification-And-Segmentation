
import open3d as o3d
import numpy as np
import os

# Threshold for detecting "floor-like" blue points
BLUE_FLOOR_THRESHOLD = 0.3  # tune depending on your colormap
MAX_DISTANCE = 2.0  # meters


def is_floor_color(colors):
    """
    Detect floor points based on blue dominance.
    Assumes RGB in [0,1].
    """
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

    # "dark blue" heuristic: high B, low R and G
    return (b > 0.5) & (b > r) & (b > g)


def annotate_cloud(ply_file):

    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    # If colors exist
    has_colors = len(pcd.colors) > 0
    colors = np.asarray(pcd.colors) if has_colors else None

    print(f"\nProcessing: {os.path.basename(ply_file)}")
    print("Select SEED POINTS (SHIFT + Left Click). Press Q when done.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Select Seeds")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()

    if len(picked) < 1:
        print("No seeds selected.")

        while True:
            choice = input("Do you want to save an empty selection? [y = save / n = retry / q = quit]: ").strip().lower()

            if choice == "y":
                # guardar máscara vacía
                labels_out = np.zeros(len(points), dtype=np.uint8)
                label_file = os.path.splitext(ply_file)[0] + "_labels.npy"
                np.save(label_file, labels_out)

                print(f"Saved EMPTY annotation: {label_file}")
                return True

            elif choice == "n":
                print("Retrying selection...")
                return annotate_cloud(ply_file)

            elif choice == "q":
                print("Skipping file.")
                return False

            else:
                print("Please type y / n / q.")

    # -----------------------------
    # GLOBAL CLUSTERING
    # -----------------------------
    labels = np.array(
        pcd.cluster_dbscan(eps=0.15, min_points=10)
    )

    seed_labels = labels[picked]
    valid_labels = set(seed_labels[seed_labels != -1])

    seed_points = points[picked]

    # compute distance from EACH point to nearest seed
    distances = np.linalg.norm(points[:, None, :] - seed_points[None, :, :], axis=2)
    min_dist = np.min(distances, axis=1)

    distance_mask = min_dist <= MAX_DISTANCE

    if len(valid_labels) == 0:
        print("No valid cluster found.")
        return False
    
    

    mask = np.isin(labels, list(valid_labels)) & distance_mask

    # -----------------------------
    # FLOOR REMOVAL (BLUE FILTER)
    # -----------------------------
    if has_colors:
        floor_mask = is_floor_color(colors)
        mask = mask & (~floor_mask)

    # -----------------------------
    # VISUAL CHECK
    # -----------------------------
    preview_pcd = o3d.geometry.PointCloud()
    preview_pcd.points = o3d.utility.Vector3dVector(points)

    preview_colors = np.full((len(points), 3), 0.7)
    preview_colors[mask] = [1, 0, 0]

    preview_pcd.colors = o3d.utility.Vector3dVector(preview_colors)

    o3d.visualization.draw_geometries(
        [preview_pcd],
        window_name="Red = Selected cluster (no floor)"
    )

    # -----------------------------
    # ACCEPT / RETRY LOOP
    # -----------------------------
    while True:
        answer = input("Accept annotation? [y/n]: ").strip().lower()

        if answer == "y":
            break

        elif answer == "n":
            print("Retrying selection...")

            # recursive retry (clean restart)
            return annotate_cloud(ply_file)

        else:
            print("Please type 'y' or 'n'.")

    # -----------------------------
    # SAVE LABELS
    # -----------------------------
    labels_out = np.zeros(len(points), dtype=np.uint8)
    labels_out[mask] = 1

    label_file = os.path.splitext(ply_file)[0] + "_labels.npy"
    np.save(label_file, labels_out)

    print(f"Saved: {label_file}")
    print(f"Selected points: {mask.sum()}")

    return True

def batch_annotate(folder_path):

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".pcd")
    ])

    print(f"Found {len(files)} point clouds.")

    # Filter only non-annotated files
    pending_files = []

    for f in files:
        pcd_file = os.path.join(folder_path, f)
        label_file = pcd_file.replace(".pcd", "_labels.npy")

        if os.path.exists(label_file):
            print(f"[SKIP] Already annotated: {f}")
            continue

        pending_files.append(f)

    print(f"\nPending annotation: {len(pending_files)} files\n")

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    for i, f in enumerate(pending_files):

        pcd_file = os.path.join(folder_path, f)

        print(f"\n[{i+1}/{len(pending_files)}] {f}")

        success = annotate_cloud(pcd_file)

        if not success:
            print("Annotation failed or skipped.")

        cmd = input("Press Enter to continue or 'q' to quit: ").strip().lower()

        if cmd == "q":
            print("Stopping batch annotation.")
            break


if __name__ == "__main__":
    batch_annotate("lidar_database/clean")