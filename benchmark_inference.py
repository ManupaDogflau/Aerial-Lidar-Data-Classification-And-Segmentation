#!/usr/bin/env python3

import argparse
import csv
import gc
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "lidar_database" / "clean"
INFERENCE_DIR = ROOT / "inference"
OPENPOINTS_DIR = ROOT / "openpoints"

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

MODELS = [
    "PointNet",
    "PointNet++",
    "DGCNN",
    "PointNeXt-S",
    "PointNeXt-XL",
    "PointVector",
]

NUM_POINTS = 4096


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def find_checkpoint(directory):

    candidates = []

    for pattern in [
        "best_model",
        "best_model.*",
        "**/best_model",
        "**/best_model.*",
    ]:
        candidates.extend(directory.glob(pattern))

    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(
            f"No best_model checkpoint found in:\n{directory}"
        )

    return candidates[-1]


def load_cloud(path):

    pcd = o3d.io.read_point_cloud(str(path))

    points = np.asarray(
        pcd.points,
        dtype=np.float32
    )

    if points.ndim != 2 or points.shape[1] < 3:
        raise RuntimeError(
            f"Invalid point cloud: {path}"
        )

    return points[:, :3]


def create_patches(points):

    patches = []

    for start in range(
        0,
        len(points),
        NUM_POINTS
    ):

        patch = points[
            start:start + NUM_POINTS
        ]

        if len(patch) == 0:
            continue

        # Last patch is padded only if necessary.
        if len(patch) < NUM_POINTS:

            indices = np.random.choice(
                len(patch),
                NUM_POINTS - len(patch),
                replace=True
            )

            patch = np.concatenate(
                [
                    patch,
                    patch[indices]
                ],
                axis=0
            )

        patches.append(
            patch.astype(np.float32)
        )

    return patches


def synchronize(device):

    if device == "cuda":
        torch.cuda.synchronize()


def cleanup_gpu():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================================
# POINTNET / POINTNET++
# ============================================================================

def import_pointnet_wrappers():

    inference_path = str(
        INFERENCE_DIR.resolve()
    )

    if inference_path not in sys.path:
        sys.path.insert(
            0,
            inference_path
        )

    from pointnet_model import PointNetModel
    from pointnet2_model import PointNet2Model

    return (
        PointNetModel,
        PointNet2Model
    )


def load_pointnet(
    model_name,
    checkpoint,
    pointnet_root,
    device
):

    PointNetModel, PointNet2Model = (
        import_pointnet_wrappers()
    )

    if model_name == "PointNet++":

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


def predict_pointnet(
    model,
    patch
):

    prediction = model.predict(
        patch
    )

    prediction = np.asarray(
        prediction
    ).reshape(-1)

    if len(prediction) != NUM_POINTS:

        raise RuntimeError(
            f"PointNet wrapper returned "
            f"{len(prediction)} predictions "
            f"for {NUM_POINTS} points."
        )

    return prediction


# ============================================================================
# OPENPOINTS
# ============================================================================

def prepare_openpoints():

    root = str(
        OPENPOINTS_DIR.resolve()
    )

    if root not in sys.path:
        sys.path.insert(
            0,
            root
        )

    imports = [
        "openpoints.models",
        "openpoints.models.backbone.dgcnn",
        "openpoints.models.backbone.pointnext",
        "openpoints.models.backbone.pointvector",
    ]

    for module in imports:

        try:
            importlib.import_module(
                module
            )
        except ImportError:
            pass


def load_openpoints_model(
    model_name,
    checkpoint,
    config,
    device
):

    prepare_openpoints()

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
        str(checkpoint),
        map_location=device
    )

    if "model_state_dict" in checkpoint_data:

        state_dict = (
            checkpoint_data[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint_data:

        state_dict = (
            checkpoint_data[
                "state_dict"
            ]
        )

    elif "model" in checkpoint_data:

        state_dict = (
            checkpoint_data["model"]
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

    points = torch.from_numpy(
        patch
    ).unsqueeze(0).to(
        device,
        non_blocking=True
    )

    xyz = points[:, :, :3].contiguous()

    # XYZ + dummy feature channel.
    ones = torch.ones(
        points.shape[0],
        points.shape[1],
        1,
        device=device
    )

    features = torch.cat(
        [
            points,
            ones
        ],
        dim=2
    )

    features = features.transpose(
        1,
        2
    ).contiguous()

    data = {
        "pos": xyz,
        "x": features,
    }

    with torch.no_grad():

        output = model(
            data
        )

    if isinstance(output, dict):

        if "logits" in output:
            output = output["logits"]

        elif "cls_logits" in output:
            output = output["cls_logits"]

        elif "pred" in output:
            output = output["pred"]

        else:

            raise RuntimeError(
                "Could not find prediction "
                "tensor in OpenPoints output."
            )

    # Common OpenPoints output:
    # [B, N, C]
    if output.ndim == 3:

        if output.shape[-1] == 2:

            prediction = output.argmax(
                dim=-1
            )

        elif output.shape[1] == 2:

            prediction = output.argmax(
                dim=1
            )

        else:

            raise RuntimeError(
                f"Unexpected output shape: "
                f"{tuple(output.shape)}"
            )

    else:

        raise RuntimeError(
            f"Unexpected output dimensions: "
            f"{tuple(output.shape)}"
        )

    return prediction.reshape(
        -1
    ).cpu().numpy()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(
    model_name,
    pointnet_root,
    device
):

    checkpoint = find_checkpoint(
        CHECKPOINT_DIRS[model_name]
    )

    print()
    print(
        f"Model:       {model_name}"
    )

    print(
        f"Checkpoint:  {checkpoint}"
    )

    if model_name in [
        "PointNet",
        "PointNet++"
    ]:

        model = load_pointnet(
            model_name,
            checkpoint,
            pointnet_root,
            device
        )

    else:

        config = CONFIG_FILES[
            model_name
        ]

        print(
            f"Config:      {config}"
        )

        model = load_openpoints_model(
            model_name,
            checkpoint,
            config,
            device
        )

    return model


# ============================================================================
# SINGLE PATCH
# ============================================================================

def run_patch(
    model_name,
    model,
    patch,
    device
):

    if model_name in [
        "PointNet",
        "PointNet++"
    ]:

        return predict_pointnet(
            model,
            patch
        )

    return predict_openpoints(
        model,
        patch,
        device
    )


# ============================================================================
# WARM-UP
# ============================================================================

def warmup(
    model_name,
    model,
    patch,
    device
):

    print()
    print(
        f"Warming up {model_name}..."
    )

    for _ in range(2):

        run_patch(
            model_name,
            model,
            patch,
            device
        )

    synchronize(
        device
    )

    print(
        "Warm-up completed."
    )


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_dataset(
    model_name,
    model,
    cloud_files,
    device
):

    print()
    print("=" * 80)
    print(
        f"RUNNING FULL DATASET: {model_name}"
    )
    print("=" * 80)

    total_inference_time = 0.0
    total_points = 0
    total_patches = 0

    cloud_results = []

    for index, cloud_file in enumerate(
        cloud_files,
        start=1
    ):

        # ------------------------------------------------------------
        # Loading is deliberately outside the timing.
        # ------------------------------------------------------------

        points = load_cloud(
            cloud_file
        )

        patches = create_patches(
            points
        )

        total_points += len(points)
        total_patches += len(patches)

        # ------------------------------------------------------------
        # Synchronize before timing so that previous CUDA work
        # cannot contaminate the measurement.
        # ------------------------------------------------------------

        synchronize(
            device
        )

        start = time.perf_counter()

        for patch in patches:

            run_patch(
                model_name,
                model,
                patch,
                device
            )

        synchronize(
            device
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        total_inference_time += elapsed

        cloud_results.append(
            {
                "file": cloud_file.name,
                "points": len(points),
                "patches": len(patches),
                "inference_time_s": elapsed,
                "points_per_second":
                    len(points) / elapsed,
            }
        )

        print(
            f"[{index:3d}/{len(cloud_files):3d}] "
            f"{cloud_file.name:<42} "
            f"{len(points):6d} pts | "
            f"{len(patches):2d} patches | "
            f"{elapsed:.4f} s"
        )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    mean_cloud_time = (
        total_inference_time
        / len(cloud_files)
    )

    points_per_second = (
        total_points
        / total_inference_time
    )

    patches_per_second = (
        total_patches
        / total_inference_time
    )

    return {
        "model": model_name,
        "clouds": len(cloud_files),
        "total_points": total_points,
        "total_patches": total_patches,
        "total_inference_time_s":
            total_inference_time,
        "mean_inference_time_per_cloud_s":
            mean_cloud_time,
        "points_per_second":
            points_per_second,
        "patches_per_second":
            patches_per_second,
        "cloud_results": cloud_results,
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    summary,
    output_file
):

    output_file = Path(
        output_file
    )

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "clouds",
                "total_points",
                "total_patches",
                "total_inference_time_s",
                "mean_inference_time_per_cloud_s",
                "points_per_second",
                "patches_per_second",
            ]
        )

        writer.writeheader()

        writer.writerow(
            {
                key: value
                for key, value in summary.items()
                if key != "cloud_results"
            }
        )

    detailed_file = (
        output_file.parent
        / (
            output_file.stem
            + "_per_cloud.csv"
        )
    )

    with open(
        detailed_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "points",
                "patches",
                "inference_time_s",
                "points_per_second",
            ]
        )

        writer.writeheader()

        writer.writerows(
            summary["cloud_results"]
        )

    print()
    print(
        f"Summary saved to: "
        f"{output_file}"
    )

    print(
        f"Per-cloud results saved to: "
        f"{detailed_file}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Standalone-style inference benchmark "
            "over the complete LiDAR dataset."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
        help="Model to benchmark."
    )

    parser.add_argument(
        "--pointnet-root",
        required=True,
        help=(
            "Root directory of the "
            "PointNet/PointNet++ implementation."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(DATASET_DIR),
        help=(
            "Directory containing the "
            "clean .pcd point clouds."
        )
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=[
            "cuda",
            "cpu"
        ]
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file."
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------

    if args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        device = "cuda"

    else:

        device = "cpu"

    print("=" * 80)
    print(
        "STANDALONE-STYLE FULL DATASET INFERENCE"
    )
    print("=" * 80)

    print(
        f"Model:    {args.model}"
    )

    print(
        f"Device:   {device}"
    )

    print(
        f"Dataset:  {args.dataset}"
    )

    # ----------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------

    dataset_dir = Path(
        args.dataset
    )

    cloud_files = sorted(
        dataset_dir.glob("*.pcd")
    )

    if not cloud_files:

        raise RuntimeError(
            f"No .pcd files found in "
            f"{dataset_dir}"
        )

    print(
        f"Point clouds: {len(cloud_files)}"
    )

    # ----------------------------------------------------------------
    # Load one cloud for warm-up
    # ----------------------------------------------------------------

    warmup_cloud = load_cloud(
        cloud_files[0]
    )

    warmup_patches = create_patches(
        warmup_cloud
    )

    warmup_patch = warmup_patches[0]

    # ----------------------------------------------------------------
    # Load ONLY the requested model
    # ----------------------------------------------------------------

    model = load_model(
        args.model,
        args.pointnet_root,
        device
    )

    # ----------------------------------------------------------------
    # Warm-up
    # ----------------------------------------------------------------

    warmup(
        args.model,
        model,
        warmup_patch,
        device
    )

    # ----------------------------------------------------------------
    # Benchmark
    # ----------------------------------------------------------------

    summary = benchmark_dataset(
        args.model,
        model,
        cloud_files,
        device
    )

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------

    if args.output is None:

        safe_name = (
            args.model
            .lower()
            .replace(
                "++",
                "pp"
            )
            .replace(
                "-",
                "_"
            )
        )

        output = (
            ROOT
            / f"inference_{safe_name}.csv"
        )

    else:

        output = Path(
            args.output
        )

    save_results(
        summary,
        output
    )

    # ----------------------------------------------------------------
    # Print final result
    # ----------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"RESULTS: {args.model}"
    )
    print("=" * 80)

    print(
        f"Clouds:             "
        f"{summary['clouds']}"
    )

    print(
        f"Total points:       "
        f"{summary['total_points']}"
    )

    print(
        f"Total patches:      "
        f"{summary['total_patches']}"
    )

    print(
        f"Total inference:    "
        f"{summary['total_inference_time_s']:.4f} s"
    )

    print(
        f"Mean cloud:         "
        f"{summary['mean_inference_time_per_cloud_s']:.4f} s"
    )

    print(
        f"Points/s:            "
        f"{summary['points_per_second']:.2f}"
    )

    print(
        f"Patches/s:           "
        f"{summary['patches_per_second']:.2f}"
    )

    print("=" * 80)

    # ----------------------------------------------------------------
    # Explicit GPU cleanup
    # ----------------------------------------------------------------

    del model

    cleanup_gpu()


if __name__ == "__main__":
    main()
