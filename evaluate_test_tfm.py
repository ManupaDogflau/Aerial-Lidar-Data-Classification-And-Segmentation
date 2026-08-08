#!/usr/bin/env python3
"""
FINAL TEST EVALUATION FOR THE TFM
=================================

Evaluates the six models used in the thesis on the HELD-OUT TEST SPLIT:

    PointNet
    PointNet++
    DGCNN
    PointNeXt-S
    PointNeXt-XL
    PointVector

The script follows the repository implementation:
    - MyLidarDataset
    - 75% train / 15% validation / 10% test
    - scene-level split with seed 42
    - 4096 points per sample
    - 20 samples per scene
    - no data augmentation for test
    - binary classes: background (0), person (1)
    - same 4-channel OpenPoints input used by the repository:
      XYZ + constant feature channel

IMPORTANT:
This is an EVALUATION script only. It does not train or modify checkpoints.

OUTPUTS (inside test_evaluation/):
    test_results.csv
    test_results_latex.tex
    test_results.md
    test_confusion_matrices.csv
    test_scene_split.txt
    test_metadata.json
    test_run.log

The generated LaTeX table is directly usable as the basis of the
results table in the TFM.

USAGE
-----

From the repository root:

    python evaluate_test_tfm.py

If the external repositories are not in the paths used in the original
training scripts:

    python evaluate_test_tfm.py \
        --pointnet_root /path/to/Pointnet_Pointnet2_pytorch \
        --openpoints_root /path/to/OpenPoints

If your dataset is elsewhere:

    python evaluate_test_tfm.py \
        --data_root /path/to/lidar_database/clean

If checkpoint names/locations differ, use:

    python evaluate_test_tfm.py --find_checkpoints

"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ============================================================
# ARGUMENTS
# ============================================================

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Evaluate all TFM segmentation models on the held-out test set."
)

parser.add_argument(
    "--data_root",
    type=str,
    default=str(ROOT / "lidar_database" / "clean"),
    help="Directory containing .pcd and *_labels.npy files."
)

parser.add_argument(
    "--pointnet_root",
    type=str,
    default=os.environ.get("POINTNET_ROOT", ""),
    help="Path to Pointnet_Pointnet2_pytorch repository."
)

parser.add_argument(
    "--openpoints_root",
    type=str,
    default=os.environ.get("OPENPOINTS_ROOT", ""),
    help="Path to OpenPoints repository."
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=8
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="0 is recommended for maximum reproducibility."
)

parser.add_argument(
    "--num_points",
    type=int,
    default=4096
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

parser.add_argument(
    "--output_dir",
    type=str,
    default=str(ROOT / "test_evaluation")
)

parser.add_argument(
    "--find_checkpoints",
    action="store_true",
    help="Only print all best_model.pth files found in the repository."
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
            candidates.extend(base.rglob("best_model.pth"))

    # Remove duplicates while preserving order.
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
            print(f"{model_name:15s}: NOT FOUND")
        else:
            print(f"{model_name:15s}: {checkpoint}")


if args.find_checkpoints:
    print_all_checkpoints()
    sys.exit(0)


# ============================================================
# PATH SETUP
# ============================================================

if args.pointnet_root:
    pointnet_root = Path(args.pointnet_root)

    if pointnet_root.exists():
        sys.path.insert(0, str(pointnet_root))
        sys.path.insert(0, str(pointnet_root / "models"))
        sys.path.insert(0, str(pointnet_root / "data_utils"))
    else:
        print(f"WARNING: PointNet root does not exist: {pointnet_root}")


if args.openpoints_root:
    openpoints_root = Path(args.openpoints_root)

    if openpoints_root.exists():
        sys.path.insert(0, str(openpoints_root))
    else:
        print(f"WARNING: OpenPoints root does not exist: {openpoints_root}")


# ============================================================
# DATASET
# ============================================================

def import_dataset():
    from data_utils.MyLidarDataset import MyLidarDataset
    return MyLidarDataset


def get_test_dataset():
    MyLidarDataset = import_dataset()

    dataset = MyLidarDataset(
        root=args.data_root,
        num_points=args.num_points,
        split="test",
        augment=False,
    )

    return dataset


def save_test_scene_split(dataset):
    path = OUTPUT_DIR / "test_scene_split.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("TEST SET SCENES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Number of scenes: {len(dataset.files)}\n")
        f.write(f"Number of samples: {len(dataset)}\n")
        f.write(f"Samples per scene: {len(dataset) // max(len(dataset.files), 1)}\n")
        f.write("\n")

        for i, filename in enumerate(dataset.files):
            f.write(f"{i+1:02d}: {filename}\n")

    return path


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state = checkpoint["model"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    model_state = model.state_dict()

    compatible = {}
    skipped = []

    for key, value in state.items():

        # Remove common DataParallel prefix.
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[7:]

        if clean_key not in model_state:
            skipped.append((key, "missing"))
            continue

        if model_state[clean_key].shape != value.shape:
            skipped.append(
                (
                    key,
                    f"{tuple(value.shape)} -> "
                    f"{tuple(model_state[clean_key].shape)}"
                )
            )
            continue

        compatible[clean_key] = value

    model_state.update(compatible)
    model.load_state_dict(model_state)

    print(
        f"    compatible weights: {len(compatible)}"
    )
    print(
        f"    skipped weights:    {len(skipped)}"
    )

    if len(compatible) == 0:
        raise RuntimeError(
            "No checkpoint weights were compatible with the model."
        )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_pointnet():
    import importlib

    model_module = importlib.import_module(
        "models.pointnet_sem_seg"
    )

    model = model_module.get_model(NUM_CLASSES)

    if hasattr(model, "apply"):
        try:
            from models.pointnet_sem_seg import inplace_relu
            model.apply(inplace_relu)
        except Exception:
            pass

    return model


def build_pointnet2():
    import importlib

    model_module = importlib.import_module(
        "models.pointnet2_sem_seg"
    )

    model = model_module.get_model(NUM_CLASSES)

    if hasattr(model, "apply"):
        try:
            from models.pointnet2_sem_seg import inplace_relu
            model.apply(inplace_relu)
        except Exception:
            pass

    return model


def build_openpoints(config_filename, backbone_module):
    """
    Build an OpenPoints BaseSeg model exactly from the YAML configuration
    used by the repository.
    """

    # Import the backbone first so the registry is populated.
    importlib = __import__("importlib")

    importlib.import_module(
        backbone_module
    )

    from openpoints.models import build_model_from_cfg
    from openpoints.utils.config import EasyConfig

    config_path = ROOT / config_filename

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    cfg = EasyConfig()
    cfg.load(str(config_path))

    model = build_model_from_cfg(cfg.model)

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

    raise ValueError(model_name)


# ============================================================
# OUTPUT HANDLING
# ============================================================

def pointnet_output_to_logits(output):

    if isinstance(output, (tuple, list)):
        output = output[0]

    # PointNet implementation returns [B*N, C] in the repository.
    if output.ndim == 2:
        return output

    if output.ndim == 3:
        if output.shape[-1] == NUM_CLASSES:
            return output.reshape(-1, NUM_CLASSES)

        if output.shape[1] == NUM_CLASSES:
            return output.transpose(1, 2).contiguous().reshape(
                -1, NUM_CLASSES
            )

    raise RuntimeError(
        f"Unexpected PointNet output shape: {tuple(output.shape)}"
    )


def openpoints_output_to_logits(output):

    if isinstance(output, dict):
        for key in ["logits", "seg_logits", "pred", "cls_logits"]:
            if key in output:
                output = output[key]
                break
        else:
            tensors = [
                value for value in output.values()
                if torch.is_tensor(value)
            ]

            if not tensors:
                raise RuntimeError(
                    "OpenPoints output dictionary contains no tensor."
                )

            output = tensors[0]

    if isinstance(output, (tuple, list)):
        tensors = [
            x for x in output
            if torch.is_tensor(x)
        ]

        if not tensors:
            raise RuntimeError(
                "OpenPoints output contains no tensor."
            )

        output = tensors[0]

    if not torch.is_tensor(output):
        raise RuntimeError(
            f"Unexpected OpenPoints output type: {type(output)}"
        )

    # Repository uses (B, C, N).
    if output.ndim == 3:
        if output.shape[1] == NUM_CLASSES:
            return output.transpose(1, 2).contiguous().reshape(
                -1, NUM_CLASSES
            )

        if output.shape[-1] == NUM_CLASSES:
            return output.reshape(-1, NUM_CLASSES)

    raise RuntimeError(
        f"Unexpected OpenPoints output shape: {tuple(output.shape)}"
    )


# ============================================================
# INFERENCE
# ============================================================

def run_model(model_name, model, loader):
    model.eval()

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    total_inference_seconds = 0.0
    total_points = 0

    with torch.no_grad():

        for points, labels in loader:

            points = points.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            if model_name in ["PointNet", "PointNet++"]:

                # Repository: B,N,C -> B,C,N.
                inputs = points.transpose(
                    2, 1
                ).contiguous()

                output = model(inputs)

                logits = pointnet_output_to_logits(
                    output
                )

            else:

                # Repository OpenPoints preprocessing:
                # XYZ + constant fourth feature.
                xyz = points[:, :, :3].contiguous()

                ones = torch.ones(
                    points.shape[0],
                    points.shape[1],
                    1,
                    device=points.device,
                    dtype=points.dtype
                )

                points4 = torch.cat(
                    [points, ones],
                    dim=2
                )

                features = points4.transpose(
                    1, 2
                ).contiguous()

                inputs = {
                    "pos": xyz,
                    "x": features
                }

                output = model(inputs)

                logits = openpoints_output_to_logits(
                    output
                )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            total_inference_seconds += (
                time.perf_counter() - t0
            )

            labels_flat = labels.reshape(-1)
            predictions = torch.argmax(
                logits,
                dim=1
            )

            valid = (
                (labels_flat >= 0)
                &
                (labels_flat < NUM_CLASSES)
            )

            labels_valid = labels_flat[valid]
            predictions_valid = predictions[valid]

            indices = (
                NUM_CLASSES * labels_valid
                + predictions_valid
            )

            batch_confusion = torch.bincount(
                indices,
                minlength=NUM_CLASSES ** 2
            ).reshape(
                NUM_CLASSES,
                NUM_CLASSES
            )

            confusion += (
                batch_confusion
                .cpu()
                .numpy()
            )

            total_points += int(
                labels_valid.numel()
            )

    return (
        confusion,
        total_inference_seconds,
        total_points
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(confusion):

    tp = np.diag(confusion).astype(np.float64)

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
        tp.sum() / max(total, 1)
    )

    iou_den = (
        tp + fp + fn
    )

    iou = np.divide(
        tp,
        iou_den,
        out=np.zeros(NUM_CLASSES),
        where=iou_den != 0
    )

    precision_den = tp + fp

    precision = np.divide(
        tp,
        precision_den,
        out=np.zeros(NUM_CLASSES),
        where=precision_den != 0
    )

    recall_den = tp + fn

    recall = np.divide(
        tp,
        recall_den,
        out=np.zeros(NUM_CLASSES),
        where=recall_den != 0
    )

    f1_den = precision + recall

    f1 = np.divide(
        2 * precision * recall,
        f1_den,
        out=np.zeros(NUM_CLASSES),
        where=f1_den != 0
    )

    return {
        "accuracy": float(accuracy),
        "miou": float(np.mean(iou)),
        "background_iou": float(iou[0]),
        "background_precision": float(precision[0]),
        "background_recall": float(recall[0]),
        "background_f1": float(f1[0]),
        "person_iou": float(iou[1]),
        "person_precision": float(precision[1]),
        "person_recall": float(recall[1]),
        "person_f1": float(f1[1]),
        "TN": int(confusion[0, 0]),
        "FP": int(confusion[0, 1]),
        "FN": int(confusion[1, 0]),
        "TP": int(confusion[1, 1]),
        "total_points": int(total),
    }


# ============================================================
# LATEX / MARKDOWN OUTPUT
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

    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Quantitative test-set performance of the evaluated semantic segmentation models.}")
    lines.append("\\label{tab:test-results}")
    lines.append("\\begin{tabular}{" + "".join(x[1] for x in columns) + "}")
    lines.append("\\hline")
    lines.append(
        "Model & Accuracy & mIoU & Person IoU & "
        "Person Precision & Person Recall & Person F1 \\\\"
    )
    lines.append("\\hline")

    for _, row in df.iterrows():

        name = str(row["Model"])

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

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


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
        + "|".join(["---"] * len(cols))
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
            + " | ".join(values)
            + " |"
        )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(args.seed)

    print("=" * 90)
    print("TFM - FINAL TEST SET EVALUATION")
    print("=" * 90)

    print(f"Repository:      {ROOT}")
    print(f"Dataset:         {args.data_root}")
    print(f"Split:           TEST")
    print(f"Random seed:     {args.seed}")
    print(f"Points/sample:   {args.num_points}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Device:          {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU:             "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = get_test_dataset()

    print(
        f"Test scenes:     {len(dataset.files)}"
    )

    print(
        f"Test samples:    {len(dataset)}"
    )

    print(
        f"Samples/scene:   "
        f"{len(dataset) // max(len(dataset.files), 1)}"
    )

    split_file = save_test_scene_split(dataset)

    print(
        f"Scene split:     {split_file}"
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

        checkpoints[model_name] = checkpoint

        if checkpoint is None:
            print(
                f"[WARNING] {model_name}: "
                f"best_model.pth not found."
            )
        else:
            print(
                f"[CHECKPOINT] {model_name}: "
                f"{checkpoint}"
            )

    print()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_results = []
    all_confusions = []

    for model_name in MODEL_CHECKPOINT_DIRS:

        checkpoint = checkpoints[model_name]

        if checkpoint is None:
            print(
                f"\nSkipping {model_name}: "
                f"checkpoint not found."
            )
            continue

        print()
        print("=" * 90)
        print(f"EVALUATING: {model_name}")
        print("=" * 90)

        # Reset the random generators before every model.
        # This guarantees that all six models see exactly the
        # same randomly sampled test patches.
        set_seed(args.seed)

        try:

            model = build_model(
                model_name
            )

            model = model.to(DEVICE)

            print(
                f"Loading checkpoint: {checkpoint}"
            )

            load_checkpoint(
                model,
                checkpoint
            )

            # Recreate the loader for each model so that the same
            # deterministic sampling sequence is used.
            set_seed(args.seed)

            test_dataset = get_test_dataset()

            loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                pin_memory=(
                    DEVICE.type == "cuda"
                ),
            )

            confusion, inference_time, total_points = run_model(
                model_name,
                model,
                loader
            )

            metrics = calculate_metrics(
                confusion
            )

            throughput = (
                total_points / inference_time
                if inference_time > 0
                else 0.0
            )

            result = {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "mIoU": metrics["miou"],
                "Background IoU": metrics["background_iou"],
                "Background Precision": metrics["background_precision"],
                "Background Recall": metrics["background_recall"],
                "Background F1": metrics["background_f1"],
                "Person IoU": metrics["person_iou"],
                "Person Precision": metrics["person_precision"],
                "Person Recall": metrics["person_recall"],
                "Person F1": metrics["person_f1"],
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TP": metrics["TP"],
                "Total Points": total_points,
                "Inference Time (s)": inference_time,
                "Points/s": throughput,
            }

            all_results.append(result)

            all_confusions.append({
                "Model": model_name,
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TP": metrics["TP"],
            })

            print()
            print("RESULT")
            print("-" * 50)
            print(
                f"Accuracy:       "
                f"{metrics['accuracy']:.4f}"
            )
            print(
                f"mIoU:           "
                f"{metrics['miou']:.4f}"
            )
            print(
                f"Person IoU:     "
                f"{metrics['person_iou']:.4f}"
            )
            print(
                f"Person Prec.:   "
                f"{metrics['person_precision']:.4f}"
            )
            print(
                f"Person Recall:  "
                f"{metrics['person_recall']:.4f}"
            )
            print(
                f"Person F1:      "
                f"{metrics['person_f1']:.4f}"
            )
            print(
                f"TN: {metrics['TN']}  "
                f"FP: {metrics['FP']}  "
                f"FN: {metrics['FN']}  "
                f"TP: {metrics['TP']}"
            )
            print(
                f"Inference:      "
                f"{inference_time:.4f} s"
            )
            print(
                f"Throughput:     "
                f"{throughput:,.0f} points/s"
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:

            print()
            print(
                f"[ERROR] {model_name}"
            )
            print(
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "\nThis model was not included in the final "
                "results because its architecture/checkpoint "
                "could not be loaded."
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not all_results:

        raise RuntimeError(
            "No model was evaluated successfully."
        )

    df = pd.DataFrame(
        all_results
    )

    # Main ranking: Person IoU, since the thesis application
    # is human segmentation.
    df = df.sort_values(
        "Person IoU",
        ascending=False
    ).reset_index(drop=True)

    confusion_df = pd.DataFrame(
        all_confusions
    )

    results_csv = (
        OUTPUT_DIR / "test_results.csv"
    )

    confusion_csv = (
        OUTPUT_DIR / "test_confusion_matrices.csv"
    )

    latex_path = (
        OUTPUT_DIR / "test_results_latex.tex"
    )

    markdown_path = (
        OUTPUT_DIR / "test_results.md"
    )

    metadata_path = (
        OUTPUT_DIR / "test_metadata.json"
    )

    df.to_csv(
        results_csv,
        index=False
    )

    confusion_df.to_csv(
        confusion_csv,
        index=False
    )

    latex = create_latex_table(
        df
    )

    latex_path.write_text(
        latex,
        encoding="utf-8"
    )

    markdown = create_markdown_table(
        df
    )

    markdown_path.write_text(
        markdown,
        encoding="utf-8"
    )

    metadata = {
        "evaluation": "held-out test set",
        "repository_root": str(ROOT),
        "data_root": str(Path(args.data_root).resolve()),
        "split": "test",
        "seed": args.seed,
        "num_points": args.num_points,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(DEVICE),
        "num_test_scenes": len(dataset.files),
        "num_test_samples": len(dataset),
        "samples_per_scene": (
            len(dataset) // max(len(dataset.files), 1)
        ),
        "augmentation": False,
        "classes": CLASS_NAMES,
        "models_evaluated": list(df["Model"]),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final console table
    # --------------------------------------------------------

    display_columns = [
        "Model",
        "Accuracy",
        "mIoU",
        "Person IoU",
        "Person Precision",
        "Person Recall",
        "Person F1",
    ]

    print()
    print()
    print("=" * 90)
    print("FINAL TEST RESULTS")
    print("=" * 90)

    print(
        df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print("=" * 90)
    print("BEST TEST MODEL")
    print("=" * 90)

    best = df.iloc[0]

    print(
        f"Model:          {best['Model']}"
    )
    print(
        f"Accuracy:       {best['Accuracy']:.4f}"
    )
    print(
        f"mIoU:           {best['mIoU']:.4f}"
    )
    print(
        f"Person IoU:     {best['Person IoU']:.4f}"
    )
    print(
        f"Person F1:      {best['Person F1']:.4f}"
    )

    print()
    print("=" * 90)
    print("FILES FOR THE TFM")
    print("=" * 90)

    print(
        f"CSV:             {results_csv}"
    )
    print(
        f"LaTeX table:     {latex_path}"
    )
    print(
        f"Markdown table:  {markdown_path}"
    )
    print(
        f"Confusion:       {confusion_csv}"
    )
    print(
        f"Scene split:     {split_file}"
    )
    print(
        f"Metadata:        {metadata_path}"
    )
    print(
        f"Log:             {LOG_PATH}"
    )

    print()
    print(
        "IMPORTANT: The values in this output are TEST-SET "
        "results. Do not replace the validation results in "
        "the thesis until these values have been checked."
    )


if __name__ == "__main__":
    main()
