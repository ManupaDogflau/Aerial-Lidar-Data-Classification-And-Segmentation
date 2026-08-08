def split_cloud_into_patches(cloud, num_points=NUM_POINTS):
    """
    Split the complete point cloud into consecutive patches of num_points.

    The last patch is padded by repeating points so that every model receives
    exactly num_points points. Predictions for the padded points are discarded
    afterwards.
    """
    n = cloud.shape[0]

    patches = []
    valid_sizes = []

    for start in range(0, n, num_points):
        patch = cloud[start:start + num_points]
        valid_size = len(patch)

        if valid_size < num_points:
            indices = np.arange(num_points) % valid_size
            patch = patch[indices]

        patches.append(patch)
        valid_sizes.append(valid_size)

    return patches, valid_sizes
