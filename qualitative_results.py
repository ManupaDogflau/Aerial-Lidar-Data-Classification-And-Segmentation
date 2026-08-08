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

The complete point cloud is processed in consecutive 4096-point patches.
Each patch is independently processed by every model and the resulting
predictions are concatenated to reconstruct a prediction for the complete
point cloud.

The PointNet and PointNet++ models are loaded through the inference wrappers
already used by the repository. The OpenPoints-based models are loaded directly
from their corresponding YAML configurations and checkpoints.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

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

    candidates = sorted(
        set(candidates),
        key=lambda p: str(p),
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
        dtype=np.float32,
    )

    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(
            f"Invalid point cloud: {path}"
        )

    return cloud[:, :3]


def create_point_patches(
    cloud,
    num_points=NUM_POINTS,
):
    """
    Split the complete point cloud into consecutive patches.

    Each patch contains exactly num_points points. If the final patch contains
    fewer points, it is temporarily padded by repeating existing points.
    The returned valid_sizes array stores the number of original points in
    each patch so that predictions for padded points can be discarded later.
    """

    n = len(cloud)

    patches = []
    valid_sizes = []

    for start in range(0, n, num_points):

        end = min(start + num_points, n)

        patch = cloud[start:end]

        valid_size = len(patch)

        if valid_size < num_points:
            if valid_size == 0:
                continue

            padding_indices = np.random.default_rng(42).choice(
                valid_size,
                num_points - valid_size,
                replace=True,
            )

            padding = patch[padding_indices]

            patch = np.concatenate(
                [patch, padding],
                axis=0,
            )

        patches.append(
            patch.astype(np.float32)
        )

        valid_sizes.append(valid_size)

    return patches, valid_sizes


def select_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device


# -------------------------------------------------------------------------
# Existing inference wrappers
# -------------------------------------------------------------------------

def import_inference_wrappers():
    """
    Import the existing PointNet and PointNet++ inference wrappers from
    the repository's inference directory.
    """

    inference_dir = str(INFERENCE_DIR.resolve())

    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)

    from pointnet_model import PointNetModel
    from pointnet2_model import PointNet2Model

    return PointNetModel, PointNet2Model


def load_pointnet_wrapper(
    checkpoint,
    pointnet_root,
    pointnet2=False,
    device="cuda",
):
    """
    Load PointNet or PointNet++ using the same inference wrapper used by
    the repository's normal inference pipeline.
    """

    PointNetModel, PointNet2Model = import_inference_wrappers()

    if pointnet2:
        model = PointNet2Model(
            checkpoint=str(checkpoint),
            pointnet_root=str(pointnet_root),
            device=device,
        )
    else:
        model = PointNetModel(
            checkpoint=str(checkpoint),
            pointnet_root=str(pointnet_root),
            device=device,
        )

    return model


def predict_pointnet_wrapper(model, cloud):
    """
    Run prediction through the existing BaseSegmentationModel API.

    The wrapper performs:
        preprocess -> forward -> postprocess

    and returns one semantic label per input point.
    """

    labels = model.predict(cloud)

    labels = np.asarray(labels).reshape(-1)

    if len(labels) != len(cloud):
        raise RuntimeError(
            f"{model.name} produced {len(labels)} predictions "
            f"for {len(cloud)} points."
        )

    return labels.astype(np.uint8)


# -------------------------------------------------------------------------
# OpenPoints models
# -------------------------------------------------------------------------

def load_openpoints_model(
    checkpoint,
    config,
    openpoints_root,
    device,
):
    """
    Load an OpenPoints segmentation model from its YAML configuration.
    """

    openpoints_root = str(
        Path(openpoints_root).resolve()
    )

    if openpoints_root not in sys.path:
        sys.path.insert(0, openpoints_root)

    models_path = os.path.join(
        openpoints_root,
        "models",
    )

    if models_path not in sys.path:
        sys.path.insert(0, models_path)

    imports = [
        "openpoints.models.backbone.dgcnn",
        "openpoints.models.backbone.pointvector",
        "openpoints.models.backbone.pointnext",
    ]

    for module_name in imports:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass

    from openpoints.models import build_model_from_cfg
    from openpoints.utils.config import EasyConfig

    cfg = EasyConfig()
    cfg.load(str(config))

    model = build_model_from_cfg(cfg.model)

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
    )

    if "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]

    elif "model" in checkpoint_data:
        state_dict = checkpoint_data["model"]

    elif "state_dict" in checkpoint_data:
        state_dict = checkpoint_data["state_dict"]

    else:
        state_dict = checkpoint_data

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def predict_openpoints(model, cloud, device):
    """
    Run inference using the same XYZ + constant fourth feature
    representation used by the repository's OpenPoints inference wrappers.
    """

    tensor = torch.from_numpy(
        cloud.astype(np.float32)
    ).unsqueeze(0).to(device)

    xyz = tensor[:, :, :3].contiguous()

    ones = torch.ones(
        tensor.shape[0],
        tensor.shape[1],
        1,
        device=device,
    )

    cloud4 = torch.cat(
        [
            tensor,
            ones,
        ],
        dim=2,
    )

    features = cloud4.transpose(
        1,
        2,
    ).contiguous()

    inputs = {
        "pos": xyz,
        "x": features,
    }

    with torch.no_grad():
        output = model(inputs)

    if isinstance(output, dict):

        if "logits" in output:
            output = output["logits"]

        elif "pred" in output:
            output = output["pred"]

        elif "cls_logits" in output:
            output = output["cls_logits"]

        else:
            raise RuntimeError(
                "Unable to identify logits in OpenPoints output. "
                f"Available keys: {list(output.keys())}"
            )

    if not torch.is_tensor(output):
        raise RuntimeError(
            f"Unexpected OpenPoints output type: {type(output)}"
        )

    # Typical OpenPoints segmentation output:
    # [B, C, N]
    #
    # Some configurations may return:
    # [B, N, C]

    if output.ndim != 3:
        raise RuntimeError(
            f"Unexpected OpenPoints output shape: {tuple(output.shape)}"
        )

    if output.shape[1] == 2 and output.shape[2] == len(cloud):

        labels = output.argmax(dim=1)

    elif output.shape[2] == 2 and output.shape[1] == len(cloud):

        labels = output.argmax(dim=2)

    else:
        raise RuntimeError(
            "OpenPoints output does not match the expected "
            f"binary segmentation shape for {len(cloud)} points: "
            f"{tuple(output.shape)}"
        )

    labels = (
        labels.squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    if len(labels) != len(cloud):
        raise RuntimeError(
            f"OpenPoints model produced {len(labels)} predictions "
            f"for {len(cloud)} points."
        )

    return labels


# -------------------------------------------------------------------------
# Patch-based inference
# -------------------------------------------------------------------------

def predict_pointnet_full_cloud(
    model,
    patches,
    valid_sizes,
):
    """
    Run PointNet/PointNet++ inference independently on every 4096-point
    patch and reconstruct the prediction for the complete cloud.
    """

    all_predictions = []

    for patch_index, (patch, valid_size) in enumerate(
        zip(patches, valid_sizes),
        start=1,
    ):
        print(
            f"    Patch {patch_index}/{len(patches)}",
            end="\r",
        )

        labels = predict_pointnet_wrapper(
            model,
            patch,
        )

        all_predictions.append(
            labels[:valid_size]
        )

    print()

    return np.concatenate(
        all_predictions,
        axis=0,
    )


def predict_openpoints_full_cloud(
    model,
    patches,
    valid_sizes,
    device,
):
    """
    Run OpenPoints inference independently on every 4096-point patch and
    reconstruct the prediction for the complete cloud.
    """

    all_predictions = []

    for patch_index, (patch, valid_size) in enumerate(
        zip(patches, valid_sizes),
        start=1,
    ):
        print(
            f"    Patch {patch_index}/{len(patches)}",
            end="\r",
        )

        labels = predict_openpoints(
            model,
            patch,
            device,
        )

        all_predictions.append(
            labels[:valid_size]
        )

    print()

    return np.concatenate(
        all_predictions,
        axis=0,
    )


# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------

def plot_point_cloud(
    ax,
    cloud,
    labels,
    title,
):
    """
    Plot a segmented point cloud.

    Class 0 = background
    Class 1 = human
    """

    background = labels == 0
    human = labels == 1

    if np.any(background):
        ax.scatter(
            cloud[background, 0],
            cloud[background, 1],
            cloud[background, 2],
            s=2,
            alpha=0.25,
            label="Background",
        )

    if np.any(human):
        ax.scatter(
            cloud[human, 0],
            cloud[human, 1],
            cloud[human, 2],
            s=5,
            alpha=0.9,
            label="Human",
        )

    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=20,
        azim=-60,
    )

    ax.legend(
        loc="upper right",
        fontsize=8,
    )

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


def save_figure(
    cloud,
    predictions,
    ground_truth,
    output_path,
    input_name,
):
    """
    Generate and save a qualitative comparison including:
        - Ground Truth
        - PointNet
        - PointNet++
        - DGCNN
        - PointNeXt-S
        - PointNeXt-XL
        - PointVector
    """

    model_names = [
        "PointNet",
        "PointNet++",
        "DGCNN",
        "PointNeXt-S",
        "PointNeXt-XL",
        "PointVector",
    ]

    fig = plt.figure(
        figsize=(18, 10)
    )

    # ------------------------------------------------------------------
    # Ground Truth
    # ------------------------------------------------------------------

    ax = fig.add_subplot(
        2,
        4,
        1,
        projection="3d",
    )

    plot_point_cloud(
        ax,
        cloud,
        ground_truth,
        "Ground Truth",
    )

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------

    for i, model_name in enumerate(model_names):

        ax = fig.add_subplot(
            2,
            4,
            i + 2,
            projection="3d",
        )

        plot_point_cloud(
            ax,
            cloud,
            predictions[model_name],
            model_name,
        )

    fig.suptitle(
        f"Qualitative semantic segmentation\n{input_name}",
        fontsize=16,
    )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative semantic segmentation "
            "results for the six evaluated models."
        )
    )

    parser.add_argument(
        "--cloud",
        default="position2_standing_0.pcd",
        help=(
            "PCD filename located in lidar_database/clean/"
        ),
    )

    parser.add_argument(
        "--output",
        default="qualitative_results.png",
        help="Output figure path.",
    )

    parser.add_argument(
        "--openpoints-root",
        default=None,
        help=(
            "Path to the OpenPoints repository. "
            "If omitted, tries ROOT/openpoints."
        ),
    )

    parser.add_argument(
        "--pointnet-root",
        required=True,
        help=(
            "Path to the PointNet/PointNet++ PyTorch repository "
            "containing models/."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device: cuda, cpu, or auto.",
    )

    args = parser.parse_args()

    device = select_device(args.device)

    print("=" * 70)
    print("QUALITATIVE SEGMENTATION")
    print("=" * 70)
    print(f"Device: {device}")

    if args.openpoints_root is None:
        openpoints_root = ROOT / "openpoints"
    else:
        openpoints_root = Path(args.openpoints_root)

    cloud_path = DATASET_DIR / args.cloud

    if not cloud_path.exists():
        raise FileNotFoundError(
            f"Point cloud not found: {cloud_path}"
        )

    cloud = load_point_cloud(cloud_path)
    
    print(f"Input cloud: {cloud_path}")
    print(f"Original points: {len(cloud)}")
    
    ground_truth = load_ground_truth(
        cloud_path,
        cloud,
    )
    
    print(
        f"Ground-truth labels: {len(ground_truth)}"
    )

    # -------------------------------------------------------------
    # Create complete-cloud patches
    # -------------------------------------------------------------

    patches, valid_sizes = create_point_patches(
        cloud,
        NUM_POINTS,
    )

    print(
        f"Patch size: {NUM_POINTS}"
    )

    print(
        f"Number of patches: {len(patches)}"
    )

    print(
        f"Points reconstructed from patches: "
        f"{sum(valid_sizes)}"
    )

    # -------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------

    checkpoints = {
        name: find_checkpoint(directory)
        for name, directory in CHECKPOINT_DIRS.items()
    }

    for name, checkpoint in checkpoints.items():
        print(f"{name:<15}: {checkpoint}")

    # -------------------------------------------------------------
    # Load PointNet / PointNet++
    # -------------------------------------------------------------

    print("\nLoading PointNet...")

    pointnet = load_pointnet_wrapper(
        checkpoint=checkpoints["PointNet"],
        pointnet_root=args.pointnet_root,
        pointnet2=False,
        device=device,
    )

    print("Loading PointNet++...")

    pointnet2 = load_pointnet_wrapper(
        checkpoint=checkpoints["PointNet++"],
        pointnet_root=args.pointnet_root,
        pointnet2=True,
        device=device,
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

        print(f"Loading {model_name}...")

        config = CONFIG_FILES[model_name]

        if not config.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config}"
            )

        openpoints_models[model_name] = (
            load_openpoints_model(
                checkpoint=checkpoints[model_name],
                config=config,
                openpoints_root=openpoints_root,
                device=device,
            )
        )

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    print("\nRunning patch-based inference...")

    predictions = {}

    print("\nPointNet")
    predictions["PointNet"] = predict_pointnet_full_cloud(
        pointnet,
        patches,
        valid_sizes,
    )

    print("PointNet++")
    predictions["PointNet++"] = predict_pointnet_full_cloud(
        pointnet2,
        patches,
        valid_sizes,
    )

    for model_name in [
        "DGCNN",
        "PointNeXt-S",
        "PointNeXt-XL",
        "PointVector",
    ]:

        print(model_name)

        predictions[model_name] = predict_openpoints_full_cloud(
            openpoints_models[model_name],
            patches,
            valid_sizes,
            device,
        )

    # -------------------------------------------------------------
    # Validate reconstructed predictions
    # -------------------------------------------------------------

    print("\nValidating predictions...")

    for model_name, labels in predictions.items():

        if len(labels) != len(cloud):
            raise RuntimeError(
                f"{model_name} produced {len(labels)} predictions "
                f"for the complete cloud containing {len(cloud)} points."
            )

        print(
            f"{model_name:<15}: "
            f"{len(labels)} predictions"
        )

    # -------------------------------------------------------------
    # Print prediction statistics
    # -------------------------------------------------------------

    print("\nPrediction statistics:")

    for model_name, labels in predictions.items():

        human_points = int(
            np.sum(labels == 1)
        )

        background_points = int(
            np.sum(labels == 0)
        )

        print(
            f"{model_name:<15}: "
            f"human={human_points:5d}, "
            f"background={background_points:5d}"
        )

    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------

    output_path = Path(args.output)

    save_figure(
        cloud=cloud,
        predictions=predictions,
        ground_truth=ground_truth,
        output_path=output_path,
        input_name=args.cloud,
    )

    print("\nFigure saved to:")
    print(output_path.resolve())

    print("\nDone.")


if __name__ == "__main__":
    main()
