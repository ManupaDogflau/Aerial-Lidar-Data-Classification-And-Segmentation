#!/usr/bin/env python3

import argparse
import csv
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


# ============================================================================
# Paths
# ============================================================================

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

MODEL_ORDER = [
    "PointNet",
    "PointNet++",
    "DGCNN",
    "PointNeXt-S",
    "PointNeXt-XL",
    "PointVector",
]


# ============================================================================
# Utilities
# ============================================================================

def select_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device


def find_checkpoint(directory):

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
        key=lambda p: str(p)
    )

    if not candidates:
        raise FileNotFoundError(
            f"No best_model checkpoint found in {directory}"
        )

    return candidates[-1]


def load_point_cloud(path):

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


def create_patches(cloud, num_points=4096):

    patches = []

    for start in range(0, len(cloud), num_points):

        end = min(
            start + num_points,
            len(cloud)
        )

        patch = cloud[start:end]

        valid_size = len(patch)

        if valid_size == 0:
            continue

        if valid_size < num_points:

            rng = np.random.default_rng(42)

            padding_indices = rng.choice(
                valid_size,
                num_points - valid_size,
                replace=True
            )

            padding = patch[padding_indices]

            patch = np.concatenate(
                [patch, padding],
                axis=0
            )

        patches.append(
            patch.astype(np.float32)
        )

    return patches


# ============================================================================
# PointNet / PointNet++
# ============================================================================

def import_inference_wrappers():

    inference_dir = str(
        INFERENCE_DIR.resolve()
    )

    if inference_dir not in sys.path:
        sys.path.insert(
            0,
            inference_dir
        )

    from pointnet_model import PointNetModel
    from pointnet2_model import PointNet2Model

    return PointNetModel, PointNet2Model


def load_pointnet(
    checkpoint,
    pointnet_root,
    pointnet2,
    device
):

    PointNetModel, PointNet2Model = (
        import_inference_wrappers()
    )

    if pointnet2:

        return PointNet2Model(
            checkpoint=str(checkpoint),
            pointnet_root=str(pointnet_root),
            device=device,
        )

    return PointNetModel(
        checkpoint=str(checkpoint),
        pointnet_root=str(pointnet_root),
        device=device,
    )


def predict_pointnet(model, patch):

    labels = model.predict(patch)

    labels = np.asarray(
        labels
    ).reshape(-1)

    if len(labels) != len(patch):

        raise RuntimeError(
            f"{model.name} produced "
            f"{len(labels)} predictions for "
            f"{len(patch)} points."
        )

    return labels


# ============================================================================
# OpenPoints
# ============================================================================

def load_openpoints_model(
    checkpoint,
    config,
    openpoints_root,
    device
):

    openpoints_root = str(
        Path(openpoints_root).resolve()
    )

    if openpoints_root not in sys.path:
        sys.path.insert(
            0,
            openpoints_root
        )

    models_path = os.path.join(
        openpoints_root,
        "models"
    )

    if models_path not in sys.path:
        sys.path.insert(
            0,
            models_path
        )

    imports = [
        "openpoints.models.backbone.dgcnn",
        "openpoints.models.backbone.pointvector",
        "openpoints.models.backbone.pointnext",
    ]

    for module_name in imports:

        try:
            importlib.import_module(
                module_name
            )
        except ImportError:
            pass

    from openpoints.models import (
        build_model_from_cfg
    )

    from openpoints.utils.config import (
        EasyConfig
    )

    cfg = EasyConfig()

    cfg.load(
        str(config)
    )

    model = build_model_from_cfg(
        cfg.model
    )

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device
    )

    if "model_state_dict" in checkpoint_data:

        state_dict = (
            checkpoint_data[
                "model_state_dict"
            ]
        )

    elif "model" in checkpoint_data:

        state_dict = (
            checkpoint_data["model"]
        )

    elif "state_dict" in checkpoint_data:

        state_dict = (
            checkpoint_data["state_dict"]
        )

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
    patch,
    device
):

    tensor = torch.from_numpy(
        patch.astype(np.float32)
    ).unsqueeze(0).to(device)

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
                "Unable to identify logits "
                "in model output."
            )

    if output.shape[1] == 2:

        labels = output.argmax(
            dim=1
        )

    elif output.shape[2] == 2:

        labels = output.argmax(
            dim=2
        )

    else:

        raise RuntimeError(
            "Unexpected output shape: "
            f"{tuple(output.shape)}"
        )

    return labels.squeeze(
        0
    ).cpu().numpy()


# ============================================================================
# Inference
# ============================================================================

def run_model_on_dataset(
    name,
    model,
    clouds,
    device,
    is_pointnet
):

    print()
    print("=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)

    total_points = 0
    total_patches = 0
    total_time = 0.0

    cloud_times = []

    for index, cloud_path in enumerate(
        clouds,
        start=1
    ):

        cloud = load_point_cloud(
            cloud_path
        )

        patches = create_patches(
            cloud,
            NUM_POINTS
        )

        start = time.perf_counter()

        for patch in patches:

            if is_pointnet:

                predict_pointnet(
                    model,
                    patch
                )

            else:

                predict_openpoints(
                    model,
                    patch,
                    device
                )

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        total_time += elapsed

        total_points += len(cloud)
        total_patches += len(patches)

        cloud_times.append(
            elapsed
        )

        print(
            f"[{index:3d}/{len(clouds):3d}] "
            f"{cloud_path.name:<40} "
            f"{len(cloud):6d} points | "
            f"{len(patches):2d} patches | "
            f"{elapsed:.4f} s"
        )

    mean_cloud_time = (
        total_time / len(clouds)
    )

    points_per_second = (
        total_points / total_time
    )

    patches_per_second = (
        total_patches / total_time
    )

    mean_points_per_cloud = (
        total_points / len(clouds)
    )

    mean_patches_per_cloud = (
        total_patches / len(clouds)
    )

    print()
    print(
        f"Total clouds:          {len(clouds)}"
    )

    print(
        f"Total original points: {total_points}"
    )

    print(
        f"Total patches:         {total_patches}"
    )

    print(
        f"Total inference time:  "
        f"{total_time:.4f} s"
    )

    print(
        f"Mean cloud time:       "
        f"{mean_cloud_time:.4f} s"
    )

    print(
        f"Points per second:     "
        f"{points_per_second:.2f}"
    )

    print(
        f"Patches per second:    "
        f"{patches_per_second:.2f}"
    )

    return {
        "model": name,
        "clouds": len(clouds),
        "total_points": total_points,
        "total_patches": total_patches,
        "total_time_s": total_time,
        "mean_cloud_time_s": mean_cloud_time,
        "mean_points_per_cloud": mean_points_per_cloud,
        "mean_patches_per_cloud": mean_patches_per_cloud,
        "points_per_second": points_per_second,
        "patches_per_second": patches_per_second,
    }


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run inference over the complete "
            "LiDAR dataset using all trained models."
        )
    )

    parser.add_argument(
        "--pointnet-root",
        required=True,
        help=(
            "Path to the PointNet/PointNet++ "
            "repository root."
        )
    )

    parser.add_argument(
        "--openpoints-root",
        default=None,
        help=(
            "Path to OpenPoints. "
            "Defaults to ./openpoints"
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(DATASET_DIR),
        help="Directory containing .pcd files."
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu"
        ]
    )

    parser.add_argument(
        "--output",
        default="full_dataset_inference.csv"
    )

    args = parser.parse_args()

    device = select_device(
        args.device
    )

    print("=" * 70)
    print(
        "FULL DATASET INFERENCE BENCHMARK"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    dataset_dir = Path(
        args.dataset
    )

    if not dataset_dir.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{dataset_dir}"
        )

    # ------------------------------------------------------------
    # Find all point clouds
    # ------------------------------------------------------------

    clouds = sorted(
        dataset_dir.glob("*.pcd")
    )

    if not clouds:

        raise FileNotFoundError(
            f"No .pcd files found in "
            f"{dataset_dir}"
        )

    print(
        f"Dataset: {dataset_dir}"
    )

    print(
        f"Point clouds found: "
        f"{len(clouds)}"
    )

    # ------------------------------------------------------------
    # OpenPoints path
    # ------------------------------------------------------------

    if args.openpoints_root is None:

        openpoints_root = (
            ROOT / "openpoints"
        )

    else:

        openpoints_root = Path(
            args.openpoints_root
        )

    # ------------------------------------------------------------
    # Find checkpoints
    # ------------------------------------------------------------

    print()
    print("Searching checkpoints...")

    checkpoints = {}

    for name, directory in (
        CHECKPOINT_DIRS.items()
    ):

        checkpoint = find_checkpoint(
            directory
        )

        checkpoints[name] = checkpoint

        print(
            f"{name:<15}: {checkpoint}"
        )

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------

    models = {}

    print()
    print("Loading models...")

    print(
        "Loading PointNet..."
    )

    models["PointNet"] = load_pointnet(
        checkpoints["PointNet"],
        args.pointnet_root,
        False,
        device
    )

    print(
        "Loading PointNet++..."
    )

    models["PointNet++"] = load_pointnet(
        checkpoints["PointNet++"],
        args.pointnet_root,
        True,
        device
    )

    for name in [
        "DGCNN",
        "PointNeXt-S",
        "PointNeXt-XL",
        "PointVector"
    ]:

        print(
            f"Loading {name}..."
        )

        models[name] = (
            load_openpoints_model(
                checkpoints[name],
                CONFIG_FILES[name],
                openpoints_root,
                device
            )
        )

    # ------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------

    print()
    print("Running one warm-up inference...")

    warmup_cloud = load_point_cloud(
        clouds[0]
    )

    warmup_patches = create_patches(
        warmup_cloud,
        NUM_POINTS
    )

    warmup_patch = warmup_patches[0]

    for name in MODEL_ORDER:

        print(
            f"Warm-up: {name}"
        )

        if name in [
            "PointNet",
            "PointNet++"
        ]:

            predict_pointnet(
                models[name],
                warmup_patch
            )

        else:

            predict_openpoints(
                models[name],
                warmup_patch,
                device
            )

    if device == "cuda":
        torch.cuda.synchronize()

    # ------------------------------------------------------------
    # Full dataset inference
    # ------------------------------------------------------------

    results = []

    for name in MODEL_ORDER:

        is_pointnet = name in [
            "PointNet",
            "PointNet++"
        ]

        result = run_model_on_dataset(
            name=name,
            model=models[name],
            clouds=clouds,
            device=device,
            is_pointnet=is_pointnet
        )

        results.append(
            result
        )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    output_path = Path(
        args.output
    )

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)

    print(
        f"{'Model':<15}"
        f"{'Clouds':>8}"
        f"{'Patches':>10}"
        f"{'Total (s)':>12}"
        f"{'Mean/cloud':>14}"
        f"{'Points/s':>15}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['model']:<15}"
            f"{result['clouds']:>8}"
            f"{result['total_patches']:>10}"
            f"{result['total_time_s']:>12.3f}"
            f"{result['mean_cloud_time_s']:>14.4f}"
            f"{result['points_per_second']:>15.1f}"
        )

    print("=" * 90)

    print()
    print(
        f"Results saved to: "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
