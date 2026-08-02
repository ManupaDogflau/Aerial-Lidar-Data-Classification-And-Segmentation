import glob
import os

import numpy as np
import open3d as o3d


class PCDDataset:

    def __init__(self, root):

        self.files = sorted(
            glob.glob(
                os.path.join(root, "*.pcd")
            )
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]

        pcd = o3d.io.read_point_cloud(file)

        cloud = np.asarray(
            pcd.points,
            dtype=np.float32
        )

        return cloud

    def __iter__(self):

        for i in range(len(self)):

            yield self[i]