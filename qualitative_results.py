#!/usr/bin/env python3

"""
Generate qualitative semantic-segmentation visualizations for the six
evaluated models.

Models:
    - PointNet
    - PointNet++
    - DGCNN
    - PointNeXt-S
    - PointNeXt-XL
    - PointVector

The same 4096-point subset is used for all models to make the qualitative
comparison visually consistent.

Expected repository structure:

Aerial-Lidar-Data-Classification-And-Segmentation/
├── inference/
├── lidar_database/
│   └── clean/
├── checkpoints_person/
├── checkpoints-pointnet2/
├── checkpoints_person-dgcnn/
├── checkpoints_person-pointnext-s/
├── checkpoints_person-pointnext-xl/
├── checkpoints_person-pointvector/
├── dgcnn.yaml
├── pointnext-s.yaml
├── pointnext-xl.yaml
└── pointvector.yaml

External dependencies:
    - PointNet/PointNet++ PyTorch repository
    - OpenPoints repository
"""

import argparse
import glob
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

INFERENCE_DIR = ROOT / "inference"
DATASET_DIR = ROOT / "lidar_database" / "clean"

CHECKPOINT_DIRS = {
    "PointNet": ROOT / "checkpoints_person",
    "PointNet++": ROOT / "checkpoints-pointnet2",
    "DGCNN": ROOT / "checkpoints_person-dgcnn",
    "PointNeXt-S": ROOT / "checkpoints_person-pointnext-s",
    "PointNeXt-XL": ROOT / "checkpoints_person-pointnext-xl",
    "PointVector": ROOT / "checkpoints_person-pointvector",
}

CONFIG_FILES = {
    "DGCNN": ROOT / "dgcnn.yaml",
    "PointNeXt-S": ROOT / "pointnext-s.yaml",
    "PointNeXt-XL": ROOT / "pointnext-xl.yaml",
    "PointVector": ROOT / "pointvector.yaml",
}


NUM_POINTS = 4096


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def find_checkpoint(directory):
    """
    Find a checkpoint named best_model, best_model.pth, etc.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {directory}"
        )

    candidates = []

    for pattern in [
        "best_model",
        "best_model.*",
        "**/best_model",
        "**/best_model.*",
    ]:
        candidates.extend(directory.glob(pattern))

    # Remove duplicates
    candidates = sorted(
        set(candidates),
        key=lambda p: str(p)
    )

    if not candidates:
        raise FileNotFoundError(
            f"No 'best_model' checkpoint found in {directory}"
        )

    if len(candidates) > 1:
        print(
            f"[WARNING] Multiple checkpoints found in {directory}. "
            f"Using: {candidates[-1]}"
        )

    return candidates[-1]


def load_point_cloud(path):
    """
    Load XYZ coordinates from a PCD file.
    """

    pcd = o3d.io.read_point_cloud(str(path))

    cloud = np.asarray(
        pcd.points,
        dtype=np.float32
    )

    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(
            f"Invalid point cloud: {path}"
        )

    return cloud[:, :3]


def sample_cloud(cloud, num_points=NUM_POINTS, seed=42):
    """
    Select exactly num_points from the cloud.

    The same random seed is used for all models so that the
    qualitative comparison uses the same points.
    """

    rng = np.random.default_rng(seed)

    n = cloud.shape[0]

    if n >= num_points:
        indices = rng.choice(
            n,
            num_points,
            replace=False
        )
    else:
        indices = rng.choice(
            n,
            num_points,
            replace=True
        )

    return cloud[indices]


def select_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device


# ---------------------------------------------------------------------
# PointNet / PointNet++
# ---------------------------------------------------------------------

def load_pointnet_model(
    checkpoint,
    pointnet_root,
    pointnet2=False,
    device="cuda"
):
    """
    Load the existing PointNet/PointNet++ implementation.
    """

    pointnet_root = str(Path(pointnet_root).resolve())

    sys.path.insert(0, pointnet_root)
    sys.path.insert(
        0,
        os.path.join(
            pointnet_root,
            "models"
        )
    )

    if pointnet2:
        module_name = "models.pointnet2_sem_seg"
    else:
        module_name = "models.pointnet_sem_seg"

    model_module = importlib.import_module(module_name)

    model = model_module.get_model(2)

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device
    )

    if "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    elif "state_dict" in checkpoint_data:
        state_dict = checkpoint_data["state_dict"]
    else:
        state_dict = checkpoint_data

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.to(device)
    model.eval()

    return model


def predict_pointnet(model, cloud, device):
    """
    PointNet / PointNet++ prediction.
    """

    tensor = torch.from_numpy(
        cloud.astype(np.float32)
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .transpose(2, 1)
        .contiguous()
        .to(device)
    )

    with torch.no_grad():

        prediction, _ = model(tensor)

        labels = prediction.argmax(
            dim=1
        )

    return (
        labels
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


# ---------------------------------------------------------------------
# OpenPoints models
# ---------------------------------------------------------------------

def load_openpoints_model(
    checkpoint,
    config,
    openpoints_root,
    device
):
    """
    Generic loader for OpenPoints-based segmentation models.

    This is used for:
        - DGCNN
        - PointNeXt-S
        - PointNeXt-XL
        - PointVector
    """

    openpoints_root = str(
        Path(openpoints_root).resolve()
    )

    sys.path.insert(
        0,
        openpoints_root
    )

    sys.path.insert(
        0,
        os.path.join(
            openpoints_root,
            "models"
        )
    )

    # Import the OpenPoints backbones used by the configurations.
    imports = [
        "openpoints.models.backbone.dgcnn",
        "openpoints.models.backbone.pointvector",
    ]

    # PointNeXt is required for PointNeXt configurations.
    try:
        imports.append(
            "openpoints.models.backbone.pointnext"
        )
    except Exception:
        pass

    for module_name in imports:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass

    from openpoints.models import build_model_from_cfg
    from openpoints.utils.config import EasyConfig

    cfg = EasyConfig()
    cfg.load(str(config))

    model = build_model_from_cfg(
        cfg.model
    )

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device
    )

    if "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data[
            "model_state_dict"
        ]
    elif "model" in checkpoint_data:
        state_dict = checkpoint_data[
            "model"
        ]
    else:
        state_dict = checkpoint_data

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model


def predict_openpoints(
    model,
    cloud,
    device
):
    """
    Run inference using the same XYZ + constant fourth feature
    representation used by the repository inference wrappers.
    """

    tensor = torch.from_numpy(
        cloud.astype(np.float32)
    ).unsqueeze(0)

    tensor = tensor.to(device)

    xyz = tensor[:, :, :3].contiguous()

    ones = torch.ones(
        tensor.shape[0],
        tensor.shape[1],
        1,
        device=device
    )

    cloud4 = torch.cat(
        [
            tensor,
            ones
        ],
        dim=2
    )

    features = cloud4.transpose(
        1,
        2
    ).contiguous()

    inputs = {
        "pos": xyz,
        "x": features
    }

    with torch.no_grad():

        prediction = model(
            inputs
        )

        prediction = prediction.transpose(
            1,
            2
        )

        labels = torch.argmax(
            prediction,
            dim=-1
        )

    return (
        labels
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_prediction(
    ax,
    cloud,
    labels,
    title
):
    """
    Plot a binary segmentation result.

    Background:
        label 0

    Human:
        label 1
    """

    background = labels == 0
    human = labels == 1

    ax.scatter(
        cloud[background, 0],
        cloud[background, 1],
        cloud[background, 2],
        s=1,
        alpha=0.20
    )

    ax.scatter(
        cloud[human, 0],
        cloud[human, 1],
        cloud[human, 2],
        s=3,
        alpha=0.90
    )

    ax.set_title(
        title,
        fontsize=11
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Equal aspect ratio
    try:
        ax.set_box_aspect(
            (
                np.ptp(cloud[:, 0]),
                np.ptp(cloud[:, 1]),
                np.ptp(cloud[:, 2])
            )
        )
    except Exception:
        pass


def generate_figure(
    cloud,
    predictions,
    output_path,
    cloud_name
):
    """
    Generate a single 2x3 figure containing all six models.
    """

    model_order = [
        "PointNet",
        "PointNet++",
        "DGCNN",
        "PointNeXt-S",
        "PointNeXt-XL",
        "PointVector",
    ]

    fig = plt.figure(
        figsize=(15, 9)
    )

    for i, model_name in enumerate(
        model_order,
        start=1
    ):

        ax = fig.add_subplot(
            2,
            3,
            i,
            projection="3d"
        )

        plot_prediction(
            ax,
            cloud,
            predictions[model_name],
            model_name
        )

        ax.view_init(
            elev=20,
            azim=-60
        )

    fig.suptitle(
        f"Qualitative semantic segmentation: {cloud_name}",
        fontsize=14
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative segmentation "
            "results for all evaluated models."
        )
    )

    parser.add_argument(
        "--pcd",
        required=True,
        help="Input PCD file."
    )

    parser.add_argument(
        "--pointnet-root",
        required=True,
        help=(
            "Root directory of the PointNet/"
            "PointNet++ PyTorch implementation."
        )
    )

    parser.add_argument(
        "--openpoints-root",
        required=True,
        help=(
            "Root directory of the OpenPoints/"
            "PointNeXt repository."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="qualitative_results",
        help="Output directory."
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu or cuda."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    device = select_device(
        args.device
    )

    print("=" * 60)
    print("QUALITATIVE SEGMENTATION")
    print("=" * 60)
    print(f"Device: {device}")

    # -------------------------------------------------------------
    # Input
    # -------------------------------------------------------------

    pcd_path = Path(args.pcd)

    if not pcd_path.exists():
        raise FileNotFoundError(
            f"PCD file not found: {pcd_path}"
        )

    cloud_original = load_point_cloud(
        pcd_path
    )

    cloud = sample_cloud(
        cloud_original,
        NUM_POINTS,
        args.seed
    )

    print(
        f"Input cloud: {pcd_path}"
    )

    print(
        f"Original points: "
        f"{len(cloud_original)}"
    )

    print(
        f"Points used for visualization: "
        f"{len(cloud)}"
    )

    # -------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------

    checkpoints = {}

    for model_name, directory in CHECKPOINT_DIRS.items():

        checkpoints[model_name] = find_checkpoint(
            directory
        )

        print(
            f"{model_name:15s}: "
            f"{checkpoints[model_name]}"
        )

    # -------------------------------------------------------------
    # Load PointNet
    # -------------------------------------------------------------

    print("\nLoading PointNet...")

    pointnet = load_pointnet_model(
        checkpoints["PointNet"],
        args.pointnet_root,
        pointnet2=False,
        device=device
    )

    # -------------------------------------------------------------
    # Load PointNet++
    # -------------------------------------------------------------

    print("Loading PointNet++...")

    pointnet2 = load_pointnet_model(
        checkpoints["PointNet++"],
        args.pointnet_root,
        pointnet2=True,
        device=device
    )

    # -------------------------------------------------------------
    # Load OpenPoints models
    # -------------------------------------------------------------

    openpoints_models = {}

    for model_name in [
        "DGCNN",
        "PointNeXt-S",
        "PointNeXt-XL",
        "PointVector",
    ]:

        print(
            f"Loading {model_name}..."
        )

        openpoints_models[model_name] = (
            load_openpoints_model(
                checkpoints[model_name],
                CONFIG_FILES[model_name],
                args.openpoints_root,
                device
            )
        )

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    predictions = {}

    print("\nRunning inference...")

    print("  PointNet")
    predictions["PointNet"] = predict_pointnet(
        pointnet,
        cloud,
        device
    )

    print("  PointNet++")
    predictions["PointNet++"] = predict_pointnet(
        pointnet2,
        cloud,
        device
    )

    for model_name in [
        "DGCNN",
        "PointNeXt-S",
        "PointNeXt-XL",
        "PointVector",
    ]:

        print(
            f"  {model_name}"
        )

        predictions[model_name] = (
            predict_openpoints(
                openpoints_models[model_name],
                cloud,
                device
            )
        )

    # -------------------------------------------------------------
    # Validate output
    # -------------------------------------------------------------

    for model_name, labels in predictions.items():

        if len(labels) != len(cloud):

            raise RuntimeError(
                f"{model_name} produced "
                f"{len(labels)} predictions for "
                f"{len(cloud)} points."
            )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"{pcd_path.stem}_qualitative.png"
    )

    generate_figure(
        cloud,
        predictions,
        output_path,
        pcd_path.name
    )

    print("\nDone.")
    print(
        f"Figure saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
