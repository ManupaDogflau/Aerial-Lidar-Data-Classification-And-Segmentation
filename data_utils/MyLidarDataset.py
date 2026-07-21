import os
import random
import numpy as np
import open3d as o3d

from torch.utils.data import Dataset


class MyLidarDataset(Dataset):

    def __init__(
        self,
        root,
        num_points=4096,
        split='train',
        augment=True
    ):

        self.root = root
        self.num_points = num_points
        self.split = split
        self.augment = augment


        files = [
            f for f in os.listdir(root)
            if f.endswith('.pcd')
        ]

        files.sort()


        # separación por escenas
        random.seed(42)
        random.shuffle(files)


        n = len(files)

        if split == 'train':
            self.files = files[:int(0.75*n)]

        elif split == 'val':
            self.files = files[int(0.75*n):int(0.9*n)]

        else:
            self.files = files[int(0.9*n):]


        print(
            split,
            "escenas:",
            len(self.files)
        )


    def __len__(self):

        # varias muestras por escena
        return len(self.files) * 20



    def load_scene(self, idx):

        filename = self.files[idx]

        pcd_path = os.path.join(
            self.root,
            filename
        )


        cloud = o3d.io.read_point_cloud(
            pcd_path
        )


        points = np.asarray(
            cloud.points
        ).astype(np.float32)


        label_file = filename.replace(
            '.pcd',
            '_labels.npy'
        )


        labels = np.load(
            os.path.join(
                self.root,
                label_file
            )
        ).astype(np.int64)


        assert len(points)==len(labels), (
            points.shape,
            labels.shape
        )


        return points, labels



    def sample_points(
        self,
        points,
        labels
    ):


        person_idx = np.where(
            labels == 1
        )[0]


        # 70% muestras forzando persona
        use_person = (
            len(person_idx)>0 and
            random.random()<0.7
        )


        if use_person:

            center_id = random.choice(
                person_idx
            )

            center = points[center_id]


            dist = np.linalg.norm(
                points-center,
                axis=1
            )


            idx = np.argsort(dist)[
                :self.num_points
            ]


        else:

            if len(points) >= self.num_points:

                idx = np.random.choice(
                    len(points),
                    self.num_points,
                    replace=False
                )

            else:

                idx = np.random.choice(
                    len(points),
                    self.num_points,
                    replace=True
                )


        sampled_points = points[idx]
        sampled_labels = labels[idx]


        return sampled_points, sampled_labels



    def augment_points(
        self,
        points
    ):


        # rotación alrededor de Z
        theta = np.random.uniform(
            0,
            2*np.pi
        )


        rot = np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta),
                    0
                ],
                [
                    np.sin(theta),
                    np.cos(theta),
                    0
                ],
                [
                    0,
                    0,
                    1
                ]
            ]
        )


        points = points.dot(rot)


        # escala
        scale = np.random.uniform(
            0.95,
            1.05
        )

        points *= scale


        # jitter
        noise = np.random.normal(
            0,
            0.005,
            points.shape
        )

        points += noise.astype(
            np.float32
        )


        # pequeño desplazamiento
        shift = np.random.uniform(
            -0.02,
            0.02,
            3
        )

        points += shift


        # dropout puntos
        if random.random() < 0.3:

            drop = np.random.rand(
                len(points)
            ) < 0.1


            points[drop] = points[0]


        return points



    def __getitem__(
        self,
        index
    ):


        scene_id = index % len(self.files)


        points, labels = self.load_scene(
            scene_id
        )


        points, labels = self.sample_points(
            points,
            labels
        )


        if self.split == 'train' and self.augment:

            points = self.augment_points(
                points
            )


        return (
            points.astype(np.float32),
            labels.astype(np.int64)
        )