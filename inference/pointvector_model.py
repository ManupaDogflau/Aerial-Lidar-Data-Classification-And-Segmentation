import os
import sys

import numpy as np
import torch

from base_model import BaseSegmentationModel


class PointVectorModel(BaseSegmentationModel):

    def __init__(
        self,
        checkpoint: str,
        config: str,
        openpoints_root: str,
        device: str = "auto",
        num_classes: int = 2,
    ):

        super().__init__(
            checkpoint=checkpoint,
            device=device,
            num_classes=num_classes
        )

        self.config = config
        self.openpoints_root = openpoints_root

        self.load()

    #################################################################

    def load(self):

        sys.path.insert(0, self.openpoints_root)

        sys.path.insert(
            0,
            os.path.join(
                self.openpoints_root,
                "models"
            )
        )

        import openpoints.models.backbone.pointvector

        from openpoints.models import (
            build_model_from_cfg
        )

        from openpoints.utils.config import (
            EasyConfig
        )

        cfg = EasyConfig()

        cfg.load(self.config)

        self.model = build_model_from_cfg(
            cfg.model
        )

        checkpoint = torch.load(
            self.checkpoint,
            map_location=self.device
        )

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        else:

            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(self.device)

        self.model.eval()

    #################################################################

    @torch.no_grad()
    def preprocess(
        self,
        cloud: np.ndarray
    ):

        cloud = torch.from_numpy(
            cloud
        ).float()

        cloud = cloud.unsqueeze(0)

        cloud = cloud.to(self.device)

        xyz = cloud[:, :, :3].contiguous()

        ones = torch.ones(
            cloud.shape[0],
            cloud.shape[1],
            1,
            device=self.device
        )

        cloud4 = torch.cat(

            [

                cloud,

                ones

            ],

            dim=2

        )

        features = cloud4.transpose(
            1,
            2
        ).contiguous()

        return {

            "pos": xyz,

            "x": features

        }

    #################################################################

    @torch.no_grad()
    def forward(self, inputs):

        return self.model(inputs)

    #################################################################

    @torch.no_grad()
    def postprocess(self, prediction):

        prediction = prediction.transpose(
            1,
            2
        )

        prediction = torch.argmax(
            prediction,
            dim=-1
        )

        return prediction.squeeze(0).cpu().numpy()

    #################################################################

    @torch.no_grad()
    def predict(
        self,
        cloud
    ):

        inputs = self.preprocess(cloud)

        pred = self.forward(inputs)

        return self.postprocess(pred)