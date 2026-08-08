#!/usr/bin/env python3

import argparse
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

    candidates = sorted(set(candidates), key=lambda p: str(p))

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

        end = min(start + num_points, len(cloud))

        patch = cloud[start:end]

        valid_size = len(patch)

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

    inference_dir = str(INFERENCE_DIR.resolve())

    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)

    from pointnet_model import PointNetModel
    from pointnet2_model import PointNet2Model

    return PointNetModel, PointNet2Model


def load_pointnet(
    checkpoint,
    pointnet_root,
    pointnet2,
    device
):

    PointNetModel, PointNet2Model = import_inference_wrappers()

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

    labels = np.asarray(labels).reshape(-1)

    if len(labels) != len(patch):

        raise RuntimeError(
            f"{model.name} produced {len(labels)} predictions "
            f"for {len(patch)} points."
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
        sys.path.insert(0, openpoints_root)

    models_path = os.path.join(
        openpoints_root,
        "models"
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

    elif "state_dict" in checkpoint_data:

        state_dict = checkpoint_data[
            "state_dict"
        ]

    else:

        state_dict = checkpoint_data

    model.load_state_dict(state_dict)

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
                "Unable to identify logits in model output."
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
            f"Unexpected output shape: "
            f"{tuple(output.shape)}"
        )

    return labels.squeeze(0).cpu().numpy()


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_model(
    name,
    model,
    patches,
    device,
    is_pointnet=False,
    warmup=2,
    repetitions=3
):

    print(f"\nBenchmarking {name}")

    # ------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------

    for patch in patches[:warmup]:

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

    # ------------------------------------------------------------
    # Timed inference
    # ------------------------------------------------------------

    times = []

    for repetition in range(repetitions):

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

        end = time.perf_counter()

        elapsed = end - start

        times.append(elapsed)

        print(
            f"  Run {repetition + 1}: "
            f"{elapsed:.4f} s"
        )

    times = np.asarray(times)

    total_points = len(patches) * NUM_POINTS

    mean_time = times.mean()
    std_time = times.std()

    time_per_patch = mean_time / len(patches)

    points_per_second = (
        total_points / mean_time
    )

    fps = (
        len(patches) / mean_time
    )

    print(
        f"  Mean: {mean_time:.4f} s"
    )

    print(
        f"  Std:  {std_time:.4f} s"
    )

    print(
        f"  Patch: {time_per_patch:.4f} s"
    )

    print(
        f"  Throughput: "
        f"{points_per_second:.1f} points/s"
    )

    print(
        f"  Patch throughput: "
        f"{fps:.2f} patches/s"
    )

    return {
        "model": name,
        "mean_time_s": mean_time,
        "std_time_s": std_time,
        "time_per_patch_s": time_per_patch,
        "points_per_second": points_per_second,
        "patches_per_second": fps,
    }


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cloud",
        default="position2_standing_0.pcd"
    )

    parser.add_argument(
        "--pointnet-root",
        required=True
    )

    parser.add_argument(
        "--openpoints-root",
        default=None
    )

    parser.add_argument(
        "--device",
        default="auto"
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2
    )

    parser.add_argument(
        "--output",
        default="inference_benchmark.csv"
    )

    args = parser.parse_args()

    device = select_device(
        args.device
    )

    print("=" * 70)
    print("INFERENCE PERFORMANCE BENCHMARK")
    print("=" * 70)

    print(
        f"Device: {device}"
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
    # Load cloud
    # ------------------------------------------------------------

    cloud_path = (
        DATASET_DIR / args.cloud
    )

    if not cloud_path.exists():

        raise FileNotFoundError(
            cloud_path
        )

    cloud = load_point_cloud(
        cloud_path
    )

    patches = create_patches(
        cloud,
        NUM_POINTS
    )

    print(
        f"\nInput cloud: {cloud_path}"
    )

    print(
        f"Original points: {len(cloud)}"
    )

    print(
        f"Number of patches: {len(patches)}"
    )

    print(
        f"Points processed: "
        f"{len(patches) * NUM_POINTS}"
    )

    # ------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------

    checkpoints = {
        name: find_checkpoint(directory)
        for name, directory
        in CHECKPOINT_DIRS.items()
    }

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------

    models = {}

    print("\nLoading PointNet...")

    models["PointNet"] = load_pointnet(
        checkpoints["PointNet"],
        args.pointnet_root,
        False,
        device
    )

    print("Loading PointNet++...")

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

        models[name] = load_openpoints_model(
            checkpoints[name],
            CONFIG_FILES[name],
            openpoints_root,
            device
        )

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    results = []

    for name in [
        "PointNet",
        "PointNet++",
        "DGCNN",
        "PointNeXt-S",
        "PointNeXt-XL",
        "PointVector"
    ]:

        is_pointnet = name in [
            "PointNet",
            "PointNet++"
        ]

        result = benchmark_model(
            name=name,
            model=models[name],
            patches=patches,
            device=device,
            is_pointnet=is_pointnet,
            warmup=args.warmup,
            repetitions=args.repetitions
        )

        results.append(result)

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    import csv

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
    # Summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"{'Model':<15}"
        f"{'Mean (s)':>12}"
        f"{'Std (s)':>12}"
        f"{'Patch (s)':>14}"
        f"{'Points/s':>15}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['model']:<15}"
            f"{result['mean_time_s']:>12.4f}"
            f"{result['std_time_s']:>12.4f}"
            f"{result['time_per_patch_s']:>14.4f}"
            f"{result['points_per_second']:>15.1f}"
        )

    print("=" * 70)

    print(
        f"\nResults saved to: "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
