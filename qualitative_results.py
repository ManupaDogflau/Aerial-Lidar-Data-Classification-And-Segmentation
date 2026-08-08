def load_ground_truth(cloud_path, cloud):
    """
    Load ground-truth semantic labels associated with the input PCD.

    The repository stores labels using the naming convention:
        <cloud_name>_labels.npy

    Returns one label per point in the original point cloud.
    """

    labels_path = cloud_path.with_name(
        cloud_path.stem + "_labels.npy"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Ground-truth labels not found: {labels_path}"
        )

    labels = np.load(labels_path)
    labels = np.asarray(labels).reshape(-1)

    if len(labels) != len(cloud):
        raise RuntimeError(
            f"Ground truth contains {len(labels)} labels "
            f"for {len(cloud)} points."
        )

    return labels.astype(np.uint8)
