import os
import sys
import importlib

import numpy as np
import torch

from base_model import BaseSegmentationModel


class PointNet2Model(BaseSegmentationModel):

    NUM_POINTS = 4096

    def __init__(
        self,
        checkpoint,
        pointnet_root,
        device="auto"
    ):

        super().__init__(checkpoint, device)

        self.pointnet_root = pointnet_root

        self.load()

    ###########################################################

    def load(self):

        sys.path.insert(0, self.pointnet_root)

        sys.path.insert(
            0,
            os.path.join(
                self.pointnet_root,
                "models"
            )
        )

        MODEL = importlib.import_module(
            "models.pointnet2_sem_seg"
        )

        self.model = MODEL.get_model(2)

        checkpoint = torch.load(
            self.checkpoint,
            map_location=self.device
        )

        if "model_state_dict" in checkpoint:

            weights = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            weights = checkpoint["state_dict"]

        else:

            weights = checkpoint

        self.model.load_state_dict(
            weights,
            strict=False
        )

        self.model.to(self.device)

        self.model.eval()

    ###########################################################

    def preprocess(
        self,
        cloud
    ):

        cloud = cloud.astype(np.float32)

        n = cloud.shape[0]

        if n > self.NUM_POINTS:

            idx = np.random.choice(
                n,
                self.NUM_POINTS,
                replace=False
            )

            cloud = cloud[idx]

        elif n < self.NUM_POINTS:

            idx = np.random.choice(
                n,
                self.NUM_POINTS,
                replace=True
            )

            cloud = cloud[idx]

        return (
            torch.from_numpy(cloud)
            .unsqueeze(0)
            .transpose(2, 1)
            .contiguous()
            .to(self.device)
        )

    ###########################################################

    @torch.no_grad()
    def forward(
        self,
        inputs
    ):

        pred, _ = self.model(inputs)

        return pred

    ###########################################################

    def postprocess(
        self,
        prediction
    ):

        prediction = prediction.squeeze(0)

        labels = prediction.argmax(dim=1)

        return (
            labels
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    ###########################################################

    @property
    def name(self):

        return "PointNet++"

    ###########################################################

    @property
    def input_size(self):

        return self.NUM_POINTS