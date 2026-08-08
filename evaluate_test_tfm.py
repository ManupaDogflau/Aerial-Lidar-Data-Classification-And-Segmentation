#!/usr/bin/env python3

"""
TEST EVALUATION USING THE EXACT INFERENCE PIPELINE FROM
qualitative_results.py

Protocol
--------
- Same held-out scene split as MyLidarDataset:
    75% train
    15% validation
    10% test
  with seed 42.

- Every point of every test scene is evaluated.

- The inference implementation is NOT duplicated here.
  It is imported directly from qualitative_results.py.

Therefore:
    qualitative visualization
and
    quantitative evaluation

use exactly the same preprocessing, model loading,
patch creation and prediction code.

Outputs
-------
test_results.csv
test_results.md
test_results_latex.tex
test_confusion_matrices.csv
test_per_scene.csv
test_scene_split.txt
test_metadata.json
test_run.log
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "test_evaluation"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATASET_DIR = (
    ROOT / "lidar_database" / "clean"
)

NUM_CLASSES = 2

CLASS_NAMES = [
    "background",
    "person",
]

MODEL_NAMES = [
    "PointNet",
    "PointNet++",
    "DGCNN",
    "PointNeXt-S",
    "PointNeXt-XL",
    "PointVector",
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Evaluate the held-out test scenes using "
        "the exact inference pipeline from "
        "qualitative_results.py."
    )
)

parser.add_argument(
    "--pointnet-root",
    required=True,
    help=(
        "Path to the PointNet/PointNet++ "
        "PyTorch repository containing models/."
    ),
)

parser.add_argument(
    "--openpoints-root",
    default=None,
    help=(
        "Path to the OpenPoints repository. "
        "If omitted, qualitative_results.py "
        "uses ROOT/openpoints."
    ),
)

parser.add_argument(
    "--data-root",
    default=str(DATASET_DIR),
    help="Directory containing the test PCD files.",
)

parser.add_argument(
    "--device",
    default="auto",
    choices=[
        "auto",
        "cuda",
        "cpu",
    ],
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=42,
)

parser.add_argument(
    "--num-points",
    type=int,
    default=4096,
    help=(
        "Must remain 4096 to match "
        "qualitative_results.py."
    ),
)

args = parser.parse_args()


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(args.seed)
np.random.seed(args.seed)


# ============================================================
# LOGGING
# ============================================================

LOG_PATH = (
    OUTPUT_DIR / "test_run.log"
)

class Tee:

    def __init__(self, path):

        self.terminal = sys.stdout

        self.file = open(
            path,
            "w",
            encoding="utf-8"
        )

    def write(self, message):

        self.terminal.write(message)

        self.file.write(message)

    def flush(self):

        self.terminal.flush()

        self.file.flush()


sys.stdout = Tee(
    LOG_PATH
)


# ============================================================
# IMPORT THE REAL INFERENCE PIPELINE
# ============================================================

"""
This is the key point of this evaluator.

We import the functions from qualitative_results.py
instead of reimplementing model inference here.
"""

from qualitative_results import (
    CHECKPOINT_DIRS,
    CONFIG_FILES,
    NUM_POINTS,

    find_checkpoint,
    load_point_cloud,
    create_point_patches,
    select_device,

    load_pointnet_wrapper,
    predict_pointnet_full_cloud,

    load_openpoints_model,
    predict_openpoints_full_cloud,

    load_ground_truth,
)


# ============================================================
# SAFETY CHECK
# ============================================================

if args.num_points != NUM_POINTS:

    raise RuntimeError(
        f"--num-points must be {NUM_POINTS} "
        f"to match qualitative_results.py. "
        f"Received {args.num_points}."
    )


# ============================================================
# DEVICE
# ============================================================

device = select_device(
    args.device
)


# ============================================================
# TEST SPLIT
# ============================================================

def get_test_scenes(
    data_root,
    seed=42
):

    data_root = Path(
        data_root
    )

    files = [
        f
        for f in data_root.iterdir()
        if f.suffix.lower() == ".pcd"
    ]

    files.sort()

    # EXACTLY the same procedure
    # as MyLidarDataset.py.

    random.seed(seed)

    random.shuffle(files)

    n = len(files)

    test_files = files[
        int(0.9 * n):
    ]

    return test_files


def save_test_split(
    test_files
):

    path = (
        OUTPUT_DIR
        / "test_scene_split.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TEST SCENE SPLIT\n"
        )

        f.write(
            "================\n\n"
        )

        f.write(
            f"Seed: {args.seed}\n"
        )

        f.write(
            f"Scenes: {len(test_files)}\n"
        )

        f.write(
            "Evaluation: all original points\n"
        )

        f.write(
            "Inference: qualitative_results.py\n\n"
        )

        for scene in test_files:

            f.write(
                f"{scene.name}\n"
            )

    return path


# ============================================================
# CONFUSION MATRIX
# ============================================================

def confusion_matrix(
    ground_truth,
    prediction
):

    gt = np.asarray(
        ground_truth
    ).reshape(-1)

    pred = np.asarray(
        prediction
    ).reshape(-1)

    if len(gt) != len(pred):

        raise RuntimeError(
            f"Ground truth has {len(gt)} "
            f"points but prediction has "
            f"{len(pred)}."
        )

    valid = (
        (gt >= 0)
        & (gt < NUM_CLASSES)
        & (pred >= 0)
        & (pred < NUM_CLASSES)
    )

    gt = gt[valid]
    pred = pred[valid]

    cm = np.bincount(
        NUM_CLASSES * gt + pred,
        minlength=NUM_CLASSES ** 2,
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    return cm

    return (
        confusion,
        total_inference_seconds,
        total_points
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    cm
):

    cm = cm.astype(
        np.float64
    )

    tp = np.diag(cm)

    fp = (
        cm.sum(axis=0)
        - tp
    )

    fn = (
        cm.sum(axis=1)
        - tp
    )

    total = cm.sum()

    accuracy = (
        tp.sum()
        / total
        if total > 0
        else 0.0
    )

    iou_den = (
        tp + fp + fn
    )

    iou = np.divide(
        tp,
        iou_den,
        out=np.zeros(
            NUM_CLASSES
        ),
        where=iou_den != 0,
    )

    precision_den = tp + fp

    precision = np.divide(
        tp,
        precision_den,
        out=np.zeros(
            NUM_CLASSES
        ),
        where=precision_den != 0,
    )

    recall_den = tp + fn

    recall = np.divide(
        tp,
        recall_den,
        out=np.zeros(
            NUM_CLASSES
        ),
        where=recall_den != 0,
    )

    f1_den = precision + recall

    f1 = np.divide(
        2 * precision * recall,
        f1_den,
        out=np.zeros(
            NUM_CLASSES
        ),
        where=f1_den != 0,
    )

    return {

        "Accuracy":
            float(accuracy),

        "mIoU":
            float(np.mean(iou)),

        "Background IoU":
            float(iou[0]),

        "Background Precision":
            float(precision[0]),

        "Background Recall":
            float(recall[0]),

        "Background F1":
            float(f1[0]),

        "Person IoU":
            float(iou[1]),

        "Person Precision":
            float(precision[1]),

        "Person Recall":
            float(recall[1]),

        "Person F1":
            float(f1[1]),

        "TN":
            int(cm[0, 0]),

        "FP":
            int(cm[0, 1]),

        "FN":
            int(cm[1, 0]),

        "TP":
            int(cm[1, 1]),

        "Total Points":
            int(total),
    }


# ============================================================
# LATEX
# ============================================================

def make_latex(df):

    lines = [

        "\\begin{table}[ht]",

        "\\centering",

        (
            "\\caption{Test-set semantic segmentation "
            "performance using the same inference pipeline "
            "as the qualitative evaluation.}"
        ),

        "\\label{tab:test-results}",

        (
            "\\begin{tabular}{lcccccc}"
        ),

        "\\hline",

        (
            "Model & Accuracy & mIoU & Person IoU & "
            "Person Precision & Person Recall & Person F1 \\\\"
        ),

        "\\hline",
    ]

    for _, row in df.iterrows():

        lines.append(
            f"{row['Model']} & "
            f"{row['Accuracy']:.4f} & "
            f"{row['mIoU']:.4f} & "
            f"{row['Person IoU']:.4f} & "
            f"{row['Person Precision']:.4f} & "
            f"{row['Person Recall']:.4f} & "
            f"{row['Person F1']:.4f} \\\\"
        )

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# MARKDOWN
# ============================================================

def make_markdown(df):

    columns = [
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
        + " | ".join(columns)
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(
            ["---"] * len(columns)
        )
        + " |"
    )

    for _, row in df.iterrows():

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Model"]),
                    f"{row['Accuracy']:.4f}",
                    f"{row['mIoU']:.4f}",
                    f"{row['Person IoU']:.4f}",
                    f"{row['Person Precision']:.4f}",
                    f"{row['Person Recall']:.4f}",
                    f"{row['Person F1']:.4f}",
                ]
            )
            + " |"
        )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "TFM TEST EVALUATION"
    )
    print(
        "USING qualitative_results.py INFERENCE"
    )
    print("=" * 90)

    print(
        f"Device:       {device}"
    )

    print(
        f"Data root:    {args.data_root}"
    )

    print(
        f"Patch size:   {NUM_POINTS}"
    )

    print(
        f"Seed:         {args.seed}"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_scenes = get_test_scenes(
        args.data_root,
        args.seed
    )

    if not test_scenes:

        raise RuntimeError(
            "No test scenes found."
        )

    print(
        f"Test scenes: {len(test_scenes)}"
    )

    for scene in test_scenes:

        print(
            f"  {scene.name}"
        )

    split_path = save_test_split(
        test_scenes
    )

    print(
        f"\nSaved split: "
        f"{split_path}"
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    print()
    print(
        "CHECKPOINTS"
    )

    checkpoints = {}

    for model_name in MODEL_NAMES:

        checkpoint = find_checkpoint(
            CHECKPOINT_DIRS[
                model_name
            ]
        )

        checkpoints[model_name] = checkpoint

        print(
            f"{model_name:<15} "
            f"{checkpoint}"
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print(
        "LOADING MODELS"
    )

    pointnet = load_pointnet_wrapper(
        checkpoint=checkpoints[
            "PointNet"
        ],
        pointnet_root=args.pointnet_root,
        pointnet2=False,
        device=device,
    )

    pointnet2 = load_pointnet_wrapper(
        checkpoint=checkpoints[
            "PointNet++"
        ],
        pointnet_root=args.pointnet_root,
        pointnet2=True,
        device=device,
    )

    openpoints_root = (
        Path(args.openpoints_root)
        if args.openpoints_root
        else ROOT / "openpoints"
    )

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

        openpoints_models[
            model_name
        ] = load_openpoints_model(
            checkpoint=checkpoints[
                model_name
            ],
            config=CONFIG_FILES[
                model_name
            ],
            openpoints_root=openpoints_root,
            device=device,
        )

    # --------------------------------------------------------
    # Global results
    # --------------------------------------------------------

    global_confusions = {
        model_name:
            np.zeros(
                (
                    NUM_CLASSES,
                    NUM_CLASSES,
                ),
                dtype=np.int64,
            )
        for model_name in MODEL_NAMES
    }

    results = []

    per_scene_results = []

    # --------------------------------------------------------
    # Scene loop
    # --------------------------------------------------------

    for scene_number, scene_path in enumerate(
        test_scenes,
        start=1
    ):

        print()
        print("=" * 90)

        print(
            f"SCENE "
            f"{scene_number}/{len(test_scenes)}: "
            f"{scene_path.name}"
        )

        print("=" * 90)

        # ----------------------------------------------------
        # Load exactly like qualitative_results.py
        # ----------------------------------------------------

        cloud = load_point_cloud(
            scene_path
        )

        ground_truth = load_ground_truth(
            scene_path,
            cloud
        )

        patches, valid_sizes = (
            create_point_patches(
                cloud,
                NUM_POINTS
            )
        )

        print(
            f"Points:  {len(cloud):,}"
        )

        print(
            f"Patches: {len(patches):,}"
        )

        print(
            f"Reconstructed points: "
            f"{sum(valid_sizes):,}"
        )

        if sum(valid_sizes) != len(cloud):

            raise RuntimeError(
                "Patch reconstruction does not "
                "contain exactly the original "
                "number of points."
            )

        # ----------------------------------------------------
        # PointNet
        # ----------------------------------------------------

        print(
            "\nPointNet"
        )

        t0 = time.perf_counter()

        pred_pointnet = (
            predict_pointnet_full_cloud(
                pointnet,
                patches,
                valid_sizes,
            )
        )

        pointnet_time = (
            time.perf_counter() - t0
        )

        # ----------------------------------------------------
        # PointNet++
        # ----------------------------------------------------

        print(
            "PointNet++"
        )

        t0 = time.perf_counter()

        pred_pointnet2 = (
            predict_pointnet_full_cloud(
                pointnet2,
                patches,
                valid_sizes,
            )
        )

        pointnet2_time = (
            time.perf_counter() - t0
        )

        # ----------------------------------------------------
        # OpenPoints
        # ----------------------------------------------------

        predictions = {

            "PointNet":
                pred_pointnet,

            "PointNet++":
                pred_pointnet2,
        }

        inference_times = {

            "PointNet":
                pointnet_time,

            "PointNet++":
                pointnet2_time,
        }

        for model_name in [
            "DGCNN",
            "PointNeXt-S",
            "PointNeXt-XL",
            "PointVector",
        ]:

            print(
                model_name
            )

            t0 = time.perf_counter()

            prediction = (
                predict_openpoints_full_cloud(
                    openpoints_models[
                        model_name
                    ],
                    patches,
                    valid_sizes,
                    device,
                )
            )

            inference_times[
                model_name
            ] = (
                time.perf_counter()
                - t0
            )

            predictions[
                model_name
            ] = prediction

        # ----------------------------------------------------
        # Validate every prediction
        # ----------------------------------------------------

        print(
            "\nValidating predictions..."
        )

        for model_name in MODEL_NAMES:

            prediction = predictions[
                model_name
            ]

            if len(prediction) != len(cloud):

                raise RuntimeError(
                    f"{model_name}: "
                    f"{len(prediction)} predictions "
                    f"for {len(cloud)} points."
                )

        # ----------------------------------------------------
        # Metrics per model / scene
        # ----------------------------------------------------

        for model_name in MODEL_NAMES:

            prediction = predictions[
                model_name
            ]

            cm = confusion_matrix(
                ground_truth,
                prediction
            )

            global_confusions[
                model_name
            ] += cm

            metrics = calculate_metrics(
                cm
            )

            row = {

                "Model":
                    model_name,

                "Scene":
                    scene_path.name,

                "Scene Points":
                    len(cloud),

                "Patches":
                    len(patches),

                "Inference Time (s)":
                    inference_times[
                        model_name
                    ],

                **metrics,
            }

            per_scene_results.append(
                row
            )

            print(
                f"{model_name:<15} "
                f"mIoU="
                f"{metrics['mIoU']:.4f}  "
                f"Person IoU="
                f"{metrics['Person IoU']:.4f}  "
                f"F1="
                f"{metrics['Person F1']:.4f}"
            )

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    print()
    print("=" * 90)
    print(
        "GLOBAL TEST RESULTS"
    )
    print("=" * 90)

    for model_name in MODEL_NAMES:

        cm = global_confusions[
            model_name
        ]

        metrics = calculate_metrics(
            cm
        )

        result = {

            "Model":
                model_name,

            **metrics,

            "Scenes":
                len(test_scenes),

        }

        results.append(
            result
        )

        print()
        print(
            model_name
        )

        print(
            f"  Accuracy:       "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"  mIoU:           "
            f"{metrics['mIoU']:.4f}"
        )

        print(
            f"  Person IoU:     "
            f"{metrics['Person IoU']:.4f}"
        )

        print(
            f"  Person Prec.:   "
            f"{metrics['Person Precision']:.4f}"
        )

        print(
            f"  Person Recall:  "
            f"{metrics['Person Recall']:.4f}"
        )

        print(
            f"  Person F1:      "
            f"{metrics['Person F1']:.4f}"
        )

        print(
            f"  TN: {metrics['TN']:,}"
        )

        print(
            f"  FP: {metrics['FP']:,}"
        )

        print(
            f"  FN: {metrics['FN']:,}"
        )

        print(
            f"  TP: {metrics['TP']:,}"
        )

        print(
            f"  Points: "
            f"{metrics['Total Points']:,}"
        )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "Person IoU",
        ascending=False
    ).reset_index(drop=True)

    scene_df = pd.DataFrame(
        per_scene_results
    )

    confusion_rows = []

    for model_name in MODEL_NAMES:

        cm = global_confusions[
            model_name
        ]

        confusion_rows.append(
            {
                "Model":
                    model_name,

                "TN":
                    int(cm[0, 0]),

                "FP":
                    int(cm[0, 1]),

                "FN":
                    int(cm[1, 0]),

                "TP":
                    int(cm[1, 1]),
            }
        )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    latex_path = (
        OUTPUT_DIR / "test_results_latex.tex"
    )

    scene_csv = (
        OUTPUT_DIR
        / "test_per_scene.csv"
    )

    confusion_csv = (
        OUTPUT_DIR
        / "test_confusion_matrices.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False
    )

    scene_df.to_csv(
        scene_csv,
        index=False
    )

    confusion_df.to_csv(
        confusion_csv,
        index=False
    )

    # ========================================================
    # SAVE MARKDOWN
    # ========================================================

    markdown = make_markdown(
        results_df
    )

    markdown_path = (
        OUTPUT_DIR
        / "test_results.md"
    )

    markdown_path.write_text(
        "# Test Results\n\n"
        "Inference pipeline: "
        "`qualitative_results.py`\n\n"
        "Evaluation: all original points "
        "from all held-out test scenes.\n\n"
        + markdown
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # SAVE LATEX
    # ========================================================

    latex_path = (
        OUTPUT_DIR
        / "test_results_latex.tex"
    )

    latex_path.write_text(
        make_latex(results_df),
        encoding="utf-8",
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "evaluation":
            "all_points_test",

        "inference_source":
            "qualitative_results.py",

        "dataset":
            str(
                Path(
                    args.data_root
                ).resolve()
            ),

        "seed":
            args.seed,

        "num_points":
            NUM_POINTS,

        "test_scenes":
            [
                scene.name
                for scene in test_scenes
            ],

        "num_test_scenes":
            len(test_scenes),

        "models":
            MODEL_NAMES,

        "classes":
            CLASS_NAMES,

        "sampling":
            "complete cloud, consecutive patches",

        "padding":
            "temporary, predictions for padded points discarded",

        "person_centered_sampling":
            False,

        "MyLidarDataset_getitem":
            False,

        "qualitative_pipeline_reused":
            True,
    }

    metadata_path = (
        OUTPUT_DIR
        / "test_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 90)
    print(
        "FINAL TABLE"
    )
    print("=" * 90)

    print(
        results_df[
            [
                "Model",
                "Accuracy",
                "mIoU",
                "Person IoU",
                "Person Precision",
                "Person Recall",
                "Person F1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(
        "Saved:"
    )

    best = df.iloc[0]

    print(
        f"  {results_csv}"
    )
    print(
        f"  {scene_csv}"
    )
    print(
        f"  {confusion_csv}"
    )
    print(
        f"  {markdown_path}"
    )
    print(
        f"  {latex_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        f"  {split_path}"
    )

    print(
        f"  {LOG_PATH}"
    )

    print()
    print("=" * 90)
    print("FILES FOR THE TFM")
    print("=" * 90)

    print(
        f"CSV:             {results_csv}"
    )
    print(
        "Quantitative predictions are generated "
        "by the same inference functions used "
        "by qualitative_results.py."
    )
    print(
        "Every original point of every test "
        "scene is evaluated."
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
        "No person-centered test sampling is used."
    )



if __name__ == "__main__":
    main()
