import os
import sys
import random
import argparse
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from data_utils.MyLidarDataset import MyLidarDataset


# ============================================================
# CONFIGURACIÓN
# ============================================================

POINTNET_ROOT = (
    "/home/manuel/Documents/GitHub/LiDAR/"
    "Pointnet_Pointnet2_pytorch"
)

sys.path.insert(0, POINTNET_ROOT)
sys.path.insert(0, os.path.join(POINTNET_ROOT, "models"))
sys.path.insert(0, os.path.join(POINTNET_ROOT, "data_utils"))


parser = argparse.ArgumentParser()


parser.add_argument(
    '--data_root',
    default=(
        '/home/manuel/Documents/GitHub/LiDAR/'
        'lidar_database/clean'
    ),
    type=str
)


parser.add_argument(
    '--checkpoint',
    default=(
        'dgcnn.pth'
    ),
    type=str
)


parser.add_argument(
    '--save_dir',
    default='./checkpoints_person-pointvector',
    type=str
)


parser.add_argument(
    '--batch_size',
    default=8,
    type=int
)


parser.add_argument(
    '--epoch',
    default=100,
    type=int
)


parser.add_argument(
    '--lr',
    default=1e-3,
    type=float
)


parser.add_argument(
    '--num_points',
    default=4096,
    type=int
)


parser.add_argument(
    '--num_workers',
    default=4,
    type=int
)


parser.add_argument(
    '--patience',
    default=20,
    type=int
)


parser.add_argument(
    '--seed',
    default=42,
    type=int
)


args = parser.parse_args()


# ============================================================
# CONFIGURACIÓN DEL PROBLEMA
# ============================================================

NUM_CLASSES = 2

# Ejemplo:
# 0 -> background
# 1 -> person
#
# Ajusta estos nombres si tus clases son diferentes.
CLASS_NAMES = [
    "background",
    "person"
]


# Peso de las clases en CrossEntropyLoss
CLASS_WEIGHTS = [
    1.0,
    10.0
]


# Si tienes etiquetas inválidas, por ejemplo 255:
# IGNORE_INDEX = 255
#
# Si no tienes etiquetas inválidas:
IGNORE_INDEX = -100


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Más reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# UTILIDADES
# ============================================================

def load_checkpoint(model, path):

    if not os.path.exists(path):

        print(
            "\nNo se ha encontrado checkpoint."
        )

        print(
            "Entrenando desde cero.\n"
        )

        return

    print(
        "\nCargando checkpoint:"
    )

    print(path)


    checkpoint = torch.load(
        path,
        map_location='cpu'
    )


    if "model" in checkpoint:
        pretrained = checkpoint["model"]
    elif "model_state_dict" in checkpoint:

        pretrained = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        pretrained = checkpoint[
            "state_dict"
        ]

    else:

        # Si el propio checkpoint es un state_dict
        pretrained = checkpoint


    model_dict = model.state_dict()

    compatible = {}

    skipped = []


    for key, value in pretrained.items():

        if key not in model_dict:

            skipped.append(
                (key, "no existe en el modelo")
            )

            continue


        if model_dict[key].shape != value.shape:

            skipped.append(
                (
                    key,
                    f"{value.shape} -> "
                    f"{model_dict[key].shape}"
                )
            )

            continue


        compatible[key] = value


    model_dict.update(compatible)

    model.load_state_dict(
        model_dict
    )

    print(
        f"Pesos cargados: "
        f"{len(compatible)}"
    )


    print(
        f"Pesos ignorados: "
        f"{len(skipped)}",
        skipped
    )


# ============================================================
# MÉTRICAS DE SEGMENTACIÓN
# ============================================================

def update_confusion_matrix(
    confusion_matrix,
    predictions,
    labels,
    num_classes
):

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)


    valid = (
        (labels >= 0) &
        (labels < num_classes)
    )


    predictions = predictions[valid]
    labels = labels[valid]


    indices = (
        num_classes * labels +
        predictions
    )


    confusion = torch.bincount(
        indices,
        minlength=num_classes ** 2
    )


    confusion = confusion.reshape(
        num_classes,
        num_classes
    )


    confusion_matrix += confusion.cpu()


def calculate_metrics(
    confusion_matrix,
    class_names
):

    confusion_matrix = (
        confusion_matrix.numpy()
    )


    true_positive = np.diag(
        confusion_matrix
    )


    false_positive = (
        confusion_matrix.sum(axis=0)
        - true_positive
    )


    false_negative = (
        confusion_matrix.sum(axis=1)
        - true_positive
    )


    total = confusion_matrix.sum()


    # Accuracy global
    accuracy = (
        true_positive.sum()
        / max(total, 1)
    )


    # IoU
    denominator_iou = (
        true_positive +
        false_positive +
        false_negative
    )


    iou = np.divide(
        true_positive,
        denominator_iou,
        out=np.zeros_like(
            true_positive,
            dtype=float
        ),
        where=denominator_iou != 0
    )


    # Precision
    precision_denominator = (
        true_positive +
        false_positive
    )


    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros_like(
            true_positive,
            dtype=float
        ),
        where=precision_denominator != 0
    )


    # Recall
    recall_denominator = (
        true_positive +
        false_negative
    )


    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.zeros_like(
            true_positive,
            dtype=float
        ),
        where=recall_denominator != 0
    )


    # F1
    f1_denominator = (
        precision + recall
    )


    f1 = np.divide(
        2 * precision * recall,
        f1_denominator,
        out=np.zeros_like(
            precision,
            dtype=float
        ),
        where=f1_denominator != 0
    )


    # mIoU
    miou = np.mean(iou)


    # Macro metrics
    macro_precision = np.mean(
        precision
    )

    macro_recall = np.mean(
        recall
    )

    macro_f1 = np.mean(
        f1
    )


    metrics = {

        "accuracy": accuracy,

        "iou": iou,

        "miou": miou,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "confusion_matrix":
            confusion_matrix

    }


    return metrics


def print_metrics(
    metrics,
    class_names
):

    print("\n==============================")
    print("MÉTRICAS DE VALIDACIÓN")
    print("==============================")

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"mIoU: "
        f"{metrics['miou']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )


    print("\nPor clase:")


    for i, class_name in enumerate(
        class_names
    ):

        print(
            f"  {class_name:15s} | "
            f"IoU: "
            f"{metrics['iou'][i]:.4f} | "
            f"Precision: "
            f"{metrics['precision'][i]:.4f} | "
            f"Recall: "
            f"{metrics['recall'][i]:.4f} | "
            f"F1: "
            f"{metrics['f1'][i]:.4f}"
        )


    print(
        "\nMatriz de confusión:"
    )

    print(
        metrics["confusion_matrix"]
    )


# ============================================================
# VALIDACIÓN
# ============================================================

@torch.no_grad()
def validate(
    classifier,
    val_loader,
    criterion,
    device
):

    classifier.eval()


    total_loss = 0.0


    confusion_matrix = torch.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=torch.int64
    )


    for points, labels in val_loader:


        points = points.to(
            device,
            non_blocking=True
        )


        labels = labels.to(
            device,
            non_blocking=True
        )


        xyz = points[:, :, :3].contiguous()

        
        
        # Canal constante
        ones = torch.ones(
            points.shape[0],
            points.shape[1],
            1,
            device=points.device,
            dtype=points.dtype
        )

        # (B, N, 4)
        points4 = torch.cat([points, ones], dim=2)
        features = points.transpose(1,2).contiguous()

        inputs = {
            "pos": xyz,
            "x": features
        }

        pred = classifier(inputs)
        
        pred = pred.transpose(1, 2).contiguous() 
        pred = pred.reshape(-1, NUM_CLASSES)



        labels = labels.view(
            -1
        )


        loss = criterion(
            pred,
            labels
        )


        total_loss += loss.item()


        predictions = torch.argmax(
            pred,
            dim=1
        )


        update_confusion_matrix(
            confusion_matrix,
            predictions,
            labels,
            NUM_CLASSES
        )


    metrics = calculate_metrics(
        confusion_matrix,
        CLASS_NAMES
    )


    mean_loss = (
        total_loss /
        max(len(val_loader), 1)
    )


    return mean_loss, metrics


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        args.seed
    )


    os.makedirs(
        args.save_dir,
        exist_ok=True
    )
    
    log_path = os.path.join(args.save_dir, "train.log")

    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", buffering=1)  # line-buffered

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout

    print("=" * 80)
    print("Inicio del entrenamiento")
    print("=" * 80)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "\nDispositivo:",
        device
    )


    # ========================================================
    # MODELO
    # ========================================================

    import openpoints.models.backbone.pointvector
    from openpoints.models import build_model_from_cfg
    from openpoints.utils.config import EasyConfig
    from openpoints.models.build import MODELS

    cfg = EasyConfig()
    cfg.load("pointvector.yaml")

    classifier = build_model_from_cfg(cfg.model)

    classifier.to(device)

    load_checkpoint(
        classifier,
        args.checkpoint
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = MyLidarDataset(
        root=args.data_root,
        num_points=args.num_points,
        split='train'
    )


    val_dataset = MyLidarDataset(
        root=args.data_root,
        num_points=args.num_points,
        split='val'
    )


    print(
        "\nTrain samples:",
        len(train_dataset)
    )


    print(
        "Val samples:",
        len(val_dataset)
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        )
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        )
    )


    # ========================================================
    # LOSS
    # ========================================================

    weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=device
    )


    criterion = nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=IGNORE_INDEX
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )


    # ========================================================
    # MIXED PRECISION
    # ========================================================

    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )


    # ========================================================
    # TRAINING
    # ========================================================

    best_miou = -1.0

    epochs_without_improvement = 0


    for epoch in range(
        args.epoch
    ):


        classifier.train()


        total_train_loss = 0.0


        train_confusion_matrix = torch.zeros(
            (
                NUM_CLASSES,
                NUM_CLASSES
            ),
            dtype=torch.int64
        )


        for batch_idx, (
            points,
            labels
        ) in enumerate(
            train_loader
        ):


            points = points.to(
                device,
                non_blocking=True
            )


            labels = labels.to(
                device,
                non_blocking=True
            )





            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):

                xyz = points[:, :, :3].contiguous()

                
                # Canal constante
                ones = torch.ones(
                    points.shape[0],
                    points.shape[1],
                    1,
                    device=points.device,
                    dtype=points.dtype
                )

                # (B, N, 4)
                points4 = torch.cat([points, ones], dim=2)
                features = points.transpose(1,2).contiguous()


                inputs = {
                    "pos": xyz,
                    "x": features
                }

                pred = classifier(inputs)

                # (B, 2, N) -> (B, N, 2)
                pred = pred.transpose(1, 2).contiguous()

                # (B, N, 2) -> (B*N, 2)
                pred = pred.view(-1, NUM_CLASSES)

                # (B, N) -> (B*N)
                labels_flat = labels.view(-1)

                loss = criterion(
                    pred,
                    labels_flat
                )


            scaler.scale(
                loss
            ).backward()


            # Evita explosiones de gradiente
            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(),
                max_norm=1.0
            )


            scaler.step(
                optimizer
            )


            scaler.update()


            total_train_loss += (
                loss.item()
            )


            predictions = torch.argmax(
                pred.detach(),
                dim=1
            )


            update_confusion_matrix(
                train_confusion_matrix,
                predictions,
                labels_flat,
                NUM_CLASSES
            )


        train_loss = (
            total_train_loss /
            max(len(train_loader), 1)
        )


        train_metrics = calculate_metrics(
            train_confusion_matrix,
            CLASS_NAMES
        )


        # ====================================================
        # VALIDACIÓN
        # ====================================================

        val_loss, val_metrics = validate(
            classifier,
            val_loader,
            criterion,
            device
        )


        # Actualizar scheduler usando mIoU
        scheduler.step(
            val_metrics["miou"]
        )


        current_lr = optimizer.param_groups[
            0
        ]["lr"]


        print("\n")
        print("=" * 70)


        print(
            f"Epoch "
            f"{epoch + 1:03d}/"
            f"{args.epoch:03d}"
        )


        print(
            f"LR: "
            f"{current_lr:.8f}"
        )


        print(
            f"Train Loss: "
            f"{train_loss:.6f}"
        )


        print(
            f"Train Accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )


        print(
            f"Val Loss: "
            f"{val_loss:.6f}"
        )


        print(
            f"Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )


        print(
            f"Val mIoU: "
            f"{val_metrics['miou']:.4f}"
        )


        print(
            f"Person IoU: "
            f"{val_metrics['iou'][1]:.4f}"
        )


        print(
            f"Person Recall: "
            f"{val_metrics['recall'][1]:.4f}"
        )


        print(
            f"Person F1: "
            f"{val_metrics['f1'][1]:.4f}"
        )


        # ====================================================
        # CHECKPOINT LATEST
        # ====================================================

        latest_path = os.path.join(
            args.save_dir,
            "latest_model.pth"
        )


        torch.save(
            {
                "epoch":
                    epoch + 1,

                "model_state_dict":
                    classifier.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_miou":
                    best_miou,

                "val_miou":
                    val_metrics["miou"],

                "val_loss":
                    val_loss
            },
            latest_path
        )


        # ====================================================
        # CHECKPOINT BEST
        # ====================================================

        if (
            val_metrics["miou"]
            > best_miou
        ):


            best_miou = (
                val_metrics["miou"]
            )


            epochs_without_improvement = 0


            best_path = os.path.join(
                args.save_dir,
                "best_model.pth"
            )


            torch.save(
                {
                    "epoch":
                        epoch + 1,

                    "model_state_dict":
                        classifier.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "best_miou":
                        best_miou,

                    "val_miou":
                        val_metrics["miou"],

                    "val_loss":
                        val_loss,

                    "metrics":
                        val_metrics
                },
                best_path
            )


            print(
                "\nNUEVO MEJOR MODELO"
            )


            print(
                f"Best mIoU: "
                f"{best_miou:.4f}"
            )


        else:

            epochs_without_improvement += 1


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= args.patience
        ):


            print(
                "\nEarly stopping."
            )


            print(
                "No mejora durante "
                f"{args.patience} épocas."
            )


            break


        # ====================================================
        # MÉTRICAS DETALLADAS
        # ====================================================

        print_metrics(
            val_metrics,
            CLASS_NAMES
        )


    print(
        "\nEntrenamiento finalizado."
    )


    print(
        f"Mejor mIoU: "
        f"{best_miou:.4f}"
    )


if __name__ == '__main__':

    main()