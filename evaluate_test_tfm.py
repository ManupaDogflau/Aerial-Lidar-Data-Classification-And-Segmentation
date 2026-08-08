#!/usr/bin/env python3
"""
FINAL TEST EVALUATION FOR THE TFM
=================================

IMPORTANT:
This evaluator evaluates EVERY REAL POINT of every held-out test scene.

It does NOT use MyLidarDataset.__getitem__(), because that method performs
random/person-centered sampling even when augment=False.

Evaluation protocol
-------------------
1. Same scene split as MyLidarDataset:
       75% train
       15% validation
       10% test
   with seed 42.

2. Every test scene is loaded completely.

3. Each scene is divided into consecutive patches of 4096 points.

4. The final patch is padded if necessary.

5. Predictions corresponding to padded points are discarded.

6. Metrics are therefore computed ONLY over real points.

Outputs
-------
test_results.csv
test_results_latex.tex
test_results.md
test_confusion_matrices.csv
test_per_scene.csv
test_scene_split.txt
test_metadata.json
test_run.log

Usage
-----
python evaluate_test_tfm.py

Optional:
python evaluate_test_tfm.py \
    --pointnet_root /path/to/Pointnet_Pointnet2_pytorch \
    --openpoints_root /path/to/OpenPoints

python evaluate_test_tfm.py \
    --data_root /path/to/lidar_database/clean
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch


# ============================================================
# ARGUMENTS
# ============================================================

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Evaluate all TFM segmentation models on every point of the held-out test scenes."
)

parser.add_argument(
    "--data_root",
    type=str,
    default=str(ROOT / "lidar_database" / "clean"),
    help="Directory containing .pcd and *_labels.npy files.",
)

parser.add_argument(
    "--pointnet_root",
    type=str,
    default=os.environ.get("POINTNET_ROOT", ""),
    help="Path to Pointnet_Pointnet2_pytorch repository.",
)

parser.add_argument(
    "--openpoints_root",
    type=str,
    default=os.environ.get("OPENPOINTS_ROOT", ""),
    help="Path to OpenPoints repository.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Number of 4096-point patches processed together.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed used for the scene split.",
)

parser.add_argument(
    "--num_points",
    type=int,
    default=4096,
    help="Number of points per model input patch.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default=str(ROOT / "test_evaluation"),
)

parser.add_argument(
    "--find_checkpoints",
    action="store_true",
    help="Only print discovered checkpoints.",
)

args = parser.parse_args()


# ============================================================
# CONSTANTS
# ============================================================

NUM_CLASSES = 2
CLASS_NAMES = ["background", "person"]

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LOGGING
# ============================================================

LOG_PATH = OUTPUT_DIR / "test_run.log"


class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


sys.stdout = Tee(LOG_PATH)


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

MODEL_CHECKPOINT_DIRS = {
    "PointNet": [
        "checkpoints_person",
    ],
    "PointNet++": [
        "checkpoints-pointnet2",
    ],
    "DGCNN": [
        "checkpoints_person-dgcnn",
    ],
    "PointNeXt-S": [
        "checkpoints_person-pointnext-s",
    ],
    "PointNeXt-XL": [
        "checkpoints_person-pointnext-xl",
    ],
    "PointVector": [
        "checkpoints_person-pointvector",
    ],
}


def find_checkpoint(model_name):
    candidates = []

    for dirname in MODEL_CHECKPOINT_DIRS[model_name]:
        base = ROOT / dirname

        candidates.extend([
            base / "best_model.pth",
            base / "checkpoints" / "best_model.pth",
        ])

        if base.exists():
            candidates.extend(
                base.rglob("best_model.pth")
            )

    unique = []
    seen = set()

    for p in candidates:
        p = Path(p)

        if p.exists() and str(p) not in seen:
            unique.append(p)
            seen.add(str(p))

    return unique[0] if unique else None


def print_all_checkpoints():
    print("\nCHECKPOINT DISCOVERY")
    print("=" * 80)

    for model_name in MODEL_CHECKPOINT_DIRS:

        checkpoint = find_checkpoint(model_name)

        if checkpoint is None:
            print(
                f"{model_name:15s}: NOT FOUND"
            )
        else:
            print(
                f"{model_name:15s}: {checkpoint}"
            )


if args.find_checkpoints:
    print_all_checkpoints()
    sys.exit(0)


# ============================================================
# PATH SETUP
# ============================================================

if args.pointnet_root:

    pointnet_root = Path(args.pointnet_root)

    if pointnet_root.exists():

        sys.path.insert(
            0,
            str(pointnet_root)
        )

        sys.path.insert(
            0,
            str(pointnet_root / "models")
        )

        sys.path.insert(
            0,
            str(pointnet_root / "data_utils")
        )

    else:

        print(
            f"WARNING: PointNet root does not exist: "
            f"{pointnet_root}"
        )


if args.openpoints_root:

    openpoints_root = Path(args.openpoints_root)

    if openpoints_root.exists():

        sys.path.insert(
            0,
            str(openpoints_root)
        )

    else:

        print(
            f"WARNING: OpenPoints root does not exist: "
            f"{openpoints_root}"
        )


# ============================================================
# TEST SCENE SPLIT
# ============================================================

def get_test_scene_files(data_root, seed=42):

    data_root = Path(data_root)

    files = sorted(
        [
            f
            for f in data_root.iterdir()
            if f.suffix.lower() == ".pcd"
        ]
    )

    if not files:
        raise RuntimeError(
            f"No .pcd files found in {data_root}"
        )

    # IMPORTANT:
    # This reproduces MyLidarDataset exactly:
    #
    # random.seed(42)
    # random.shuffle(files)
    #
    # followed by:
    # train = files[:int(0.75*n)]
    # val   = files[int(0.75*n):int(0.9*n)]
    # test  = files[int(0.9*n):]

    files = [str(f) for f in files]

    random.seed(seed)
    random.shuffle(files)

    n = len(files)

    test_files = files[
        int(0.9 * n):
    ]

    return [
        Path(f)
        for f in test_files
    ]


def save_test_scene_split(test_files):

    path = OUTPUT_DIR / "test_scene_split.txt"

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TEST SET SCENES - ALL POINT EVALUATION\n"
        )

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            f"Number of scenes: {len(test_files)}\n"
        )

        f.write(
            f"Patch size: {args.num_points}\n"
        )

        f.write(
            "Evaluation: every real point\n"
        )

        f.write("\n")

        for i, filename in enumerate(
            test_files
        ):

            f.write(
                f"{i + 1:02d}: "
                f"{filename.name}\n"
            )

    return path


# ============================================================
# SCENE LOADING
# ============================================================

def load_scene(scene_path):

    pcd = o3d.io.read_point_cloud(
        str(scene_path)
    )

    points = np.asarray(
        pcd.points,
        dtype=np.float32
    )

    if (
        points.ndim != 2
        or points.shape[1] < 3
    ):
        raise RuntimeError(
            f"Invalid point cloud: "
            f"{scene_path}"
        )

    points = points[:, :3]

    label_path = scene_path.with_name(
        scene_path.stem + "_labels.npy"
    )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Labels not found:\n"
            f"{label_path}"
        )

    labels = np.load(
        label_path
    ).astype(np.int64)

    if len(points) != len(labels):

        raise RuntimeError(
            f"Point/label mismatch for "
            f"{scene_path.name}: "
            f"{len(points)} points vs "
            f"{len(labels)} labels"
        )

    invalid = (
        (labels < 0)
        | (labels >= NUM_CLASSES)
    )

    if invalid.any():

        raise RuntimeError(
            f"Invalid labels in "
            f"{scene_path.name}: "
            f"{np.unique(labels[invalid])}"
        )

    return points, labels


# ============================================================
# PATCH CREATION
# ============================================================

def create_scene_patches(
    points,
    labels,
    num_points
):
    """
    Divide a complete scene into consecutive patches.

    IMPORTANT:
    The final patch is padded, but the returned valid_count
    indicates how many points are real.

    Therefore padded points are NEVER included in metrics.
    """

    patches = []
    patch_labels = []
    valid_counts = []

    n = len(points)

    for start in range(
        0,
        n,
        num_points
    ):

        end = min(
            start + num_points,
            n
        )

        patch = points[start:end]
        patch_label = labels[start:end]

        valid_count = len(patch)

        if valid_count == 0:
            continue

        if valid_count < num_points:

            # Deterministic padding.
            #
            # We do NOT use the random padding from
            # benchmark_inference.py because the padded
            # points are discarded from evaluation anyway.
            #
            # Repeating the last real point avoids adding
            # randomness to the test evaluation.

            pad_count = (
                num_points
                - valid_count
            )

            if valid_count == 1:

                pad_indices = np.zeros(
                    pad_count,
                    dtype=np.int64
                )

            else:

                pad_indices = np.arange(
                    pad_count
                ) % valid_count

            patch = np.concatenate(
                [
                    patch,
                    patch[pad_indices]
                ],
                axis=0
            )

            patch_label = np.concatenate(
                [
                    patch_label,
                    patch_label[pad_indices]
                ],
                axis=0
            )

        patches.append(
            patch.astype(np.float32)
        )

        patch_labels.append(
            patch_label.astype(np.int64)
        )

        valid_counts.append(
            valid_count
        )

    return (
        patches,
        patch_labels,
        valid_counts
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state = checkpoint[
                "state_dict"
            ]

        elif (
            "model" in checkpoint
            and isinstance(
                checkpoint["model"],
                dict
            )
        ):

            state = checkpoint["model"]

        else:

            state = checkpoint

    else:

        state = checkpoint

    model_state = model.state_dict()

    compatible = {}
    skipped = []

    for key, value in state.items():

        clean_key = key

        if clean_key.startswith(
            "module."
        ):

            clean_key = clean_key[7:]

        if clean_key not in model_state:

            skipped.append(
                (key, "missing")
            )

            continue

        if (
            model_state[clean_key].shape
            != value.shape
        ):

            skipped.append(
                (
                    key,
                    (
                        f"{tuple(value.shape)} "
                        f"-> "
                        f"{tuple(model_state[clean_key].shape)}"
                    )
                )
            )

            continue

        compatible[clean_key] = value

    if len(compatible) == 0:

        raise RuntimeError(
            "No checkpoint weights were "
            "compatible with the model."
        )

    model_state.update(
        compatible
    )

    model.load_state_dict(
        model_state
    )

    print(
        f"    compatible weights: "
        f"{len(compatible)}"
    )

    print(
        f"    skipped weights:    "
        f"{len(skipped)}"
    )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_pointnet():

    import importlib

    model_module = importlib.import_module(
        "models.pointnet_sem_seg"
    )

    model = model_module.get_model(
        NUM_CLASSES
    )

    if hasattr(model, "apply"):

        try:

            from models.pointnet_sem_seg import (
                inplace_relu
            )

            model.apply(
                inplace_relu
            )

        except Exception:
            pass

    return model


def build_pointnet2():

    import importlib

    model_module = importlib.import_module(
        "models.pointnet2_sem_seg"
    )

    model = model_module.get_model(
        NUM_CLASSES
    )

    if hasattr(model, "apply"):

        try:

            from models.pointnet2_sem_seg import (
                inplace_relu
            )

            model.apply(
                inplace_relu
            )

        except Exception:
            pass

    return model


def build_openpoints(
    config_filename,
    backbone_module
):

    import importlib

    importlib.import_module(
        backbone_module
    )

    from openpoints.models import (
        build_model_from_cfg
    )

    from openpoints.utils.config import (
        EasyConfig
    )

    config_path = (
        ROOT / config_filename
    )

    if not config_path.exists():

        raise FileNotFoundError(
            f"Configuration not found: "
            f"{config_path}"
        )

    cfg = EasyConfig()

    cfg.load(
        str(config_path)
    )

    model = build_model_from_cfg(
        cfg.model
    )

    return model


def build_model(model_name):

    if model_name == "PointNet":

        return build_pointnet()

    if model_name == "PointNet++":

        return build_pointnet2()

    if model_name == "DGCNN":

        return build_openpoints(
            "dgcnn.yaml",
            "openpoints.models.backbone.dgcnn"
        )

    if model_name == "PointNeXt-S":

        return build_openpoints(
            "pointnext-s.yaml",
            "openpoints.models.backbone.pointnext"
        )

    if model_name == "PointNeXt-XL":

        return build_openpoints(
            "pointnext-xl.yaml",
            "openpoints.models.backbone.pointnext"
        )

    if model_name == "PointVector":

        return build_openpoints(
            "pointvector.yaml",
            "openpoints.models.backbone.pointvector"
        )

    raise ValueError(
        model_name
    )


# ============================================================
# MODEL OUTPUT HANDLING
# ============================================================

def pointnet_output_to_logits(
    output,
    batch_size,
    num_points
):

    if isinstance(
        output,
        (tuple, list)
    ):

        output = output[0]

    if not torch.is_tensor(output):

        raise RuntimeError(
            "PointNet output is not a tensor."
        )

    # [B*N, C]
    if output.ndim == 2:

        if (
            output.shape[0]
            != batch_size * num_points
        ):

            raise RuntimeError(
                f"Unexpected PointNet "
                f"output shape: "
                f"{tuple(output.shape)}"
            )

        return output

    # [B, N, C]
    if output.ndim == 3:

        if output.shape[0] != batch_size:

            raise RuntimeError(
                f"Unexpected PointNet "
                f"batch dimension: "
                f"{tuple(output.shape)}"
            )

        if (
            output.shape[1] == num_points
            and output.shape[2] == NUM_CLASSES
        ):

            return output.reshape(
                -1,
                NUM_CLASSES
            )

        # [B, C, N]
        if (
            output.shape[1] == NUM_CLASSES
            and output.shape[2] == num_points
        ):

            return (
                output.transpose(
                    1,
                    2
                )
                .contiguous()
                .reshape(
                    -1,
                    NUM_CLASSES
                )
            )

    raise RuntimeError(
        f"Unexpected PointNet "
        f"output shape: "
        f"{tuple(output.shape)}"
    )


def openpoints_output_to_logits(
    output,
    batch_size,
    num_points
):

    if isinstance(
        output,
        dict
    ):

        for key in [
            "logits",
            "seg_logits",
            "pred",
            "cls_logits",
        ]:

            if key in output:

                output = output[key]
                break

        else:

            tensors = [
                value
                for value in output.values()
                if torch.is_tensor(value)
            ]

            if not tensors:

                raise RuntimeError(
                    "OpenPoints output "
                    "dictionary contains "
                    "no tensor."
                )

            output = tensors[0]

    if isinstance(
        output,
        (tuple, list)
    ):

        tensors = [
            x
            for x in output
            if torch.is_tensor(x)
        ]

        if not tensors:

            raise RuntimeError(
                "OpenPoints output contains "
                "no tensor."
            )

        output = tensors[0]

    if not torch.is_tensor(output):

        raise RuntimeError(
            f"Unexpected OpenPoints output "
            f"type: {type(output)}"
        )

    if output.ndim != 3:

        raise RuntimeError(
            f"Unexpected OpenPoints output "
            f"shape: {tuple(output.shape)}"
        )

    # [B, C, N]
    if (
        output.shape[0] == batch_size
        and output.shape[1] == NUM_CLASSES
        and output.shape[2] == num_points
    ):

        return (
            output.transpose(
                1,
                2
            )
            .contiguous()
            .reshape(
                -1,
                NUM_CLASSES
            )
        )

    # [B, N, C]
    if (
        output.shape[0] == batch_size
        and output.shape[1] == num_points
        and output.shape[2] == NUM_CLASSES
    ):

        return output.reshape(
            -1,
            NUM_CLASSES
        )

    raise RuntimeError(
        f"Unexpected OpenPoints output "
        f"shape: {tuple(output.shape)}"
    )


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(
    model_name,
    model,
    patches
):

    points = torch.from_numpy(
        np.stack(patches)
    ).to(
        DEVICE,
        non_blocking=True
    )

    batch_size = points.shape[0]
    num_points = points.shape[1]

    if model_name in [
        "PointNet",
        "PointNet++",
    ]:

        # Repository convention:
        # B,N,C -> B,C,N

        inputs = points.transpose(
            2,
            1
        ).contiguous()

        output = model(
            inputs
        )

        logits = pointnet_output_to_logits(
            output,
            batch_size,
            num_points
        )

    else:

        # OpenPoints:
        # XYZ + constant feature channel

        xyz = points[
            :, :, :3
        ].contiguous()

        ones = torch.ones(
            batch_size,
            num_points,
            1,
            device=points.device,
            dtype=points.dtype
        )

        points4 = torch.cat(
            [
                points,
                ones
            ],
            dim=2
        )

        features = points4.transpose(
            1,
            2
        ).contiguous()

        inputs = {
            "pos": xyz,
            "x": features,
        }

        output = model(
            inputs
        )

        logits = openpoints_output_to_logits(
            output,
            batch_size,
            num_points
        )

    predictions = torch.argmax(
        logits,
        dim=1
    )

    predictions = predictions.reshape(
        batch_size,
        num_points
    )

    return predictions.cpu().numpy()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def update_confusion(
    confusion,
    labels,
    predictions
):

    labels = np.asarray(
        labels
    ).reshape(-1)

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    valid = (
        (labels >= 0)
        & (labels < NUM_CLASSES)
        & (predictions >= 0)
        & (predictions < NUM_CLASSES)
    )

    labels = labels[valid]
    predictions = predictions[valid]

    indices = (
        NUM_CLASSES * labels
        + predictions
    )

    batch_confusion = np.bincount(
        indices,
        minlength=NUM_CLASSES ** 2
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES
    )

    confusion += batch_confusion


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    confusion
):

    tp = np.diag(
        confusion
    ).astype(
        np.float64
    )

    fp = (
        confusion.sum(axis=0)
        - tp
    )

    fn = (
        confusion.sum(axis=1)
        - tp
    )

    total = confusion.sum()

    accuracy = (
        tp.sum()
        / max(total, 1)
    )

    iou_den = (
        tp
        + fp
        + fn
    )

    iou = np.divide(
        tp,
        iou_den,
        out=np.zeros(
            NUM_CLASSES,
            dtype=np.float64
        ),
        where=iou_den != 0
    )

    precision_den = (
        tp + fp
    )

    precision = np.divide(
        tp,
        precision_den,
        out=np.zeros(
            NUM_CLASSES,
            dtype=np.float64
        ),
        where=precision_den != 0
    )

    recall_den = (
        tp + fn
    )

    recall = np.divide(
        tp,
        recall_den,
        out=np.zeros(
            NUM_CLASSES,
            dtype=np.float64
        ),
        where=recall_den != 0
    )

    f1_den = (
        precision + recall
    )

    f1 = np.divide(
        2 * precision * recall,
        f1_den,
        out=np.zeros(
            NUM_CLASSES,
            dtype=np.float64
        ),
        where=f1_den != 0
    )

    return {
        "accuracy": float(accuracy),

        "miou": float(
            np.mean(iou)
        ),

        "background_iou": float(
            iou[0]
        ),

        "background_precision": float(
            precision[0]
        ),

        "background_recall": float(
            recall[0]
        ),

        "background_f1": float(
            f1[0]
        ),

        "person_iou": float(
            iou[1]
        ),

        "person_precision": float(
            precision[1]
        ),

        "person_recall": float(
            recall[1]
        ),

        "person_f1": float(
            f1[1]
        ),

        "TN": int(
            confusion[0, 0]
        ),

        "FP": int(
            confusion[0, 1]
        ),

        "FN": int(
            confusion[1, 0]
        ),

        "TP": int(
            confusion[1, 1]
        ),

        "total_points": int(
            total
        ),
    }


# ============================================================
# SCENE EVALUATION
# ============================================================

def evaluate_scene(
    model_name,
    model,
    scene_path
):

    points, labels = load_scene(
        scene_path
    )

    (
        patches,
        patch_labels,
        valid_counts
    ) = create_scene_patches(
        points,
        labels,
        args.num_points
    )

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    inference_time = 0.0

    total_real_points = 0

    total_patches = len(
        patches
    )

    for start in range(
        0,
        total_patches,
        args.batch_size
    ):

        batch_patches = patches[
            start:
            start + args.batch_size
        ]

        batch_labels = patch_labels[
            start:
            start + args.batch_size
        ]

        batch_valid_counts = valid_counts[
            start:
            start + args.batch_size
        ]

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        predictions = predict_batch(
            model_name,
            model,
            batch_patches
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        inference_time += (
            time.perf_counter()
            - t0
        )

        # IMPORTANT:
        # Remove predictions belonging to padded points.
        #
        # This is the key difference from the old evaluator.

        for i, valid_count in enumerate(
            batch_valid_counts
        ):

            real_labels = batch_labels[
                i
            ][:valid_count]

            real_predictions = predictions[
                i
            ][:valid_count]

            update_confusion(
                confusion,
                real_labels,
                real_predictions
            )

            total_real_points += (
                valid_count
            )

    metrics = calculate_metrics(
        confusion
    )

    metrics.update(
        {
            "Scene": scene_path.name,
            "Scene Points": len(points),
            "Patches": total_patches,
            "Inference Time (s)": inference_time,
            "Points/s": (
                len(points)
                / inference_time
                if inference_time > 0
                else 0.0
            ),
        }
    )

    return (
        confusion,
        metrics
    )


# ============================================================
# LATEX
# ============================================================

def create_latex_table(df):

    columns = [
        ("Model", "l"),
        ("Accuracy", "c"),
        ("mIoU", "c"),
        ("Person IoU", "c"),
        ("Person Precision", "c"),
        ("Person Recall", "c"),
        ("Person F1", "c"),
    ]

    lines = []

    lines.append(
        "\\begin{table}[ht]"
    )

    lines.append(
        "\\centering"
    )

    lines.append(
        "\\caption{Quantitative test-set performance of the evaluated semantic segmentation models using all points from the held-out test scenes.}"
    )

    lines.append(
        "\\label{tab:test-results}"
    )

    lines.append(
        "\\begin{tabular}{"
        + "".join(
            x[1]
            for x in columns
        )
        + "}"
    )

    lines.append(
        "\\hline"
    )

    lines.append(
        "Model & Accuracy & mIoU & Person IoU & "
        "Person Precision & Person Recall & Person F1 \\\\"
    )

    lines.append(
        "\\hline"
    )

    for _, row in df.iterrows():

        name = str(
            row["Model"]
        )

        values = [
            f"{row['Accuracy']:.4f}",
            f"{row['mIoU']:.4f}",
            f"{row['Person IoU']:.4f}",
            f"{row['Person Precision']:.4f}",
            f"{row['Person Recall']:.4f}",
            f"{row['Person F1']:.4f}",
        ]

        lines.append(
            name
            + " & "
            + " & ".join(values)
            + " \\\\"
        )

    lines.append(
        "\\hline"
    )

    lines.append(
        "\\end{tabular}"
    )

    lines.append(
        "\\end{table}"
    )

    return "\n".join(
        lines
    )


# ============================================================
# MARKDOWN
# ============================================================

def create_markdown_table(df):

    cols = [
        "Model",
        "Accuracy",
        "mIoU",
        "Person IoU",
        "Person Precision",
        "Person Recall",
        "Person F1",
    ]

    lines = []

    lines.append(
        "| "
        + " | ".join(cols)
        + " |"
    )

    lines.append(
        "|"
        + "|".join(
            ["---"] * len(cols)
        )
        + "|"
    )

    for _, row in df.iterrows():

        values = [
            row["Model"],
            f"{row['Accuracy']:.4f}",
            f"{row['mIoU']:.4f}",
            f"{row['Person IoU']:.4f}",
            f"{row['Person Precision']:.4f}",
            f"{row['Person Recall']:.4f}",
            f"{row['Person F1']:.4f}",
        ]

        lines.append(
            "| "
            + " | ".join(
                values
            )
            + " |"
        )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        args.seed
    )

    print("=" * 90)
    print(
        "TFM - FINAL TEST SET EVALUATION"
    )
    print(
        "ALL REAL POINTS / SCENE LEVEL"
    )
    print("=" * 90)

    print(
        f"Repository:      {ROOT}"
    )

    print(
        f"Dataset:         {args.data_root}"
    )

    print(
        f"Split:            TEST"
    )

    print(
        f"Random seed:     {args.seed}"
    )

    print(
        f"Points/patch:     {args.num_points}"
    )

    print(
        f"Batch size:       {args.batch_size}"
    )

    print(
        f"Device:           {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU:              "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    # --------------------------------------------------------
    # Test scenes
    # --------------------------------------------------------

    test_files = get_test_scene_files(
        args.data_root,
        args.seed
    )

    print(
        f"Test scenes:      "
        f"{len(test_files)}"
    )

    for i, scene in enumerate(
        test_files,
        start=1
    ):

        print(
            f"  {i:02d}: "
            f"{scene.name}"
        )

    split_file = save_test_scene_split(
        test_files
    )

    print()
    print(
        f"Scene split:      "
        f"{split_file}"
    )

    print()
    print(
        "Evaluation protocol:"
    )
    print(
        "  - complete test scenes"
    )
    print(
        "  - consecutive 4096-point patches"
    )
    print(
        "  - deterministic padding of final patch"
    )
    print(
        "  - padded points excluded from metrics"
    )

    print()

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    checkpoints = {}

    for model_name in MODEL_CHECKPOINT_DIRS:

        checkpoint = find_checkpoint(
            model_name
        )

        checkpoints[
            model_name
        ] = checkpoint

        if checkpoint is None:

            print(
                f"[WARNING] "
                f"{model_name}: "
                f"best_model.pth not found."
            )

        else:

            print(
                f"[CHECKPOINT] "
                f"{model_name}: "
                f"{checkpoint}"
            )

    print()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "evaluation_type":
            "all_real_points",
        "scene_level":
            True,
        "num_test_scenes":
            len(test_files),
        "test_scenes":
            [x.name for x in test_files],
        "num_points_per_patch":
            args.num_points,
        "batch_size":
            args.batch_size,
        "seed":
            args.seed,
        "device":
            str(DEVICE),
        "classes":
            CLASS_NAMES,
        "padding_points_excluded":
            True,
        "random_person_centered_sampling":
            False,
        "dataset_getitem_used":
            False,
    }

    metadata_path = (
        OUTPUT_DIR
        / "test_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_results = []
    all_confusions = []
    all_scene_results = []

    for model_name in MODEL_CHECKPOINT_DIRS:

        checkpoint = checkpoints[
            model_name
        ]

        if checkpoint is None:

            print(
                f"\nSkipping "
                f"{model_name}: "
                f"checkpoint not found."
            )

            continue

        print()
        print("=" * 90)
        print(
            f"EVALUATING: "
            f"{model_name}"
        )
        print("=" * 90)

        set_seed(
            args.seed
        )

        try:

            # ------------------------------------------------
            # Build model
            # ------------------------------------------------

            model = build_model(
                model_name
            )

            model = model.to(
                DEVICE
            )

            model.eval()

            print(
                f"Loading checkpoint: "
                f"{checkpoint}"
            )

            load_checkpoint(
                model,
                checkpoint
            )

            model.eval()

            # ------------------------------------------------
            # Global test confusion
            # ------------------------------------------------

            global_confusion = np.zeros(
                (
                    NUM_CLASSES,
                    NUM_CLASSES
                ),
                dtype=np.int64
            )

            total_inference_time = 0.0
            total_points = 0
            total_patches = 0

            # ------------------------------------------------
            # Scene loop
            # ------------------------------------------------

            for scene_number, scene_path in enumerate(
                test_files,
                start=1
            ):

                print()
                print(
                    f"[{scene_number:02d}/"
                    f"{len(test_files):02d}] "
                    f"{scene_path.name}"
                )

                (
                    scene_confusion,
                    scene_metrics
                ) = evaluate_scene(
                    model_name,
                    model,
                    scene_path
                )

                global_confusion += (
                    scene_confusion
                )

                total_inference_time += (
                    scene_metrics[
                        "Inference Time (s)"
                    ]
                )

                total_points += (
                    scene_metrics[
                        "Scene Points"
                    ]
                )

                total_patches += (
                    scene_metrics[
                        "Patches"
                    ]
                )

                scene_row = {
                    "Model":
                        model_name,
                    **scene_metrics,
                }

                all_scene_results.append(
                    scene_row
                )

                print(
                    f"    Points:       "
                    f"{scene_metrics['Scene Points']:,}"
                )

                print(
                    f"    Patches:      "
                    f"{scene_metrics['Patches']:,}"
                )

                print(
                    f"    Person IoU:   "
                    f"{scene_metrics['person_iou']:.4f}"
                )

                print(
                    f"    Person F1:    "
                    f"{scene_metrics['person_f1']:.4f}"
                )

                print(
                    f"    Person Recall:"
                    f" "
                    f"{scene_metrics['person_recall']:.4f}"
                )

            # ------------------------------------------------
            # Global metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                global_confusion
            )

            throughput = (
                total_points
                / total_inference_time
                if total_inference_time > 0
                else 0.0
            )

            result = {
                "Model":
                    model_name,

                "Accuracy":
                    metrics["accuracy"],

                "mIoU":
                    metrics["miou"],

                "Background IoU":
                    metrics["background_iou"],

                "Background Precision":
                    metrics["background_precision"],

                "Background Recall":
                    metrics["background_recall"],

                "Background F1":
                    metrics["background_f1"],

                "Person IoU":
                    metrics["person_iou"],

                "Person Precision":
                    metrics["person_precision"],

                "Person Recall":
                    metrics["person_recall"],

                "Person F1":
                    metrics["person_f1"],

                "TN":
                    metrics["TN"],

                "FP":
                    metrics["FP"],

                "FN":
                    metrics["FN"],

                "TP":
                    metrics["TP"],

                "Total Points":
                    total_points,

                "Total Patches":
                    total_patches,

                "Inference Time (s)":
                    total_inference_time,

                "Points/s":
                    throughput,
            }

            all_results.append(
                result
            )

            all_confusions.append(
                {
                    "Model":
                        model_name,

                    "TN":
                        metrics["TN"],

                    "FP":
                        metrics["FP"],

                    "FN":
                        metrics["FN"],

                    "TP":
                        metrics["TP"],
                }
            )

            # ------------------------------------------------
            # Print global result
            # ------------------------------------------------

            print()
            print(
                "GLOBAL RESULT"
            )

            print(
                "-" * 60
            )

            print(
                f"Scenes:          "
                f"{len(test_files)}"
            )

            print(
                f"Real points:     "
                f"{total_points:,}"
            )

            print(
                f"Patches:         "
                f"{total_patches:,}"
            )

            print(
                f"Accuracy:        "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"mIoU:            "
                f"{metrics['miou']:.4f}"
            )

            print(
                f"Person IoU:      "
                f"{metrics['person_iou']:.4f}"
            )

            print(
                f"Person Precision: "
                f"{metrics['person_precision']:.4f}"
            )

            print(
                f"Person Recall:   "
                f"{metrics['person_recall']:.4f}"
            )

            print(
                f"Person F1:       "
                f"{metrics['person_f1']:.4f}"
            )

            print(
                f"TN: {metrics['TN']:,}  "
                f"FP: {metrics['FP']:,}  "
                f"FN: {metrics['FN']:,}  "
                f"TP: {metrics['TP']:,}"
            )

            print(
                f"Inference:       "
                f"{total_inference_time:.4f} s"
            )

            print(
                f"Throughput:      "
                f"{throughput:,.0f} points/s"
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:

            print()
            print(
                f"[ERROR] "
                f"{model_name}"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "\nThis model was not included "
                "in the final results."
            )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not all_results:

        raise RuntimeError(
            "No model was evaluated successfully."
        )

    # --------------------------------------------------------
    # Global results
    # --------------------------------------------------------

    df = pd.DataFrame(
        all_results
    )

    # Primary ranking:
    # Person IoU
    df = df.sort_values(
        "Person IoU",
        ascending=False
    ).reset_index(
        drop=True
    )

    confusion_df = pd.DataFrame(
        all_confusions
    )

    scene_df = pd.DataFrame(
        all_scene_results
    )

    # --------------------------------------------------------
    # Save global CSV
    # --------------------------------------------------------

    results_csv = (
        OUTPUT_DIR
        / "test_results.csv"
    )

    df.to_csv(
        results_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save confusion matrices
    # --------------------------------------------------------

    confusion_csv = (
        OUTPUT_DIR
        / "test_confusion_matrices.csv"
    )

    confusion_df.to_csv(
        confusion_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save per-scene metrics
    # --------------------------------------------------------

    scene_csv = (
        OUTPUT_DIR
        / "test_per_scene.csv"
    )

    scene_df.to_csv(
        scene_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save LaTeX
    # --------------------------------------------------------

    latex = create_latex_table(
        df
    )

    latex_path = (
        OUTPUT_DIR
        / "test_results_latex.tex"
    )

    with open(
        latex_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            latex
        )

    # --------------------------------------------------------
    # Save Markdown
    # --------------------------------------------------------

    markdown = create_markdown_table(
        df
    )

    markdown_path = (
        OUTPUT_DIR
        / "test_results.md"
    )

    with open(
        markdown_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# TFM Test Results\n\n"
        )

        f.write(
            "Evaluation protocol: "
            "all real points from all held-out "
            "test scenes.\n\n"
        )

        f.write(
            markdown
        )

        f.write(
            "\n"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "FINAL TEST RESULTS"
    )
    print("=" * 90)

    display_columns = [
        "Model",
        "Accuracy",
        "mIoU",
        "Person IoU",
        "Person Precision",
        "Person Recall",
        "Person F1",
    ]

    print(
        df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()
    print(
        f"Results CSV:       "
        f"{results_csv}"
    )

    print(
        f"Confusion CSV:     "
        f"{confusion_csv}"
    )

    print(
        f"Per-scene CSV:     "
        f"{scene_csv}"
    )

    print(
        f"LaTeX:              "
        f"{latex_path}"
    )

    print(
        f"Markdown:           "
        f"{markdown_path}"
    )

    print(
        f"Metadata:           "
        f"{metadata_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The reported metrics contain "
        "ONLY real points from the held-out "
        "test scenes."
    )

    print(
        "No person-centered random sampling "
        "was used during evaluation."
    )

    print(
        "No padded points were included "
        "in the metrics."
    )

    print("=" * 90)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
