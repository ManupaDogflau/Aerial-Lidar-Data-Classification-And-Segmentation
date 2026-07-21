import os
import sys
import random
import importlib
import argparse

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


sys.path.insert(
    0,
    POINTNET_ROOT
)


sys.path.insert(
    0,
    os.path.join(
        POINTNET_ROOT,
        "models"
    )
)


sys.path.insert(
    0,
    os.path.join(
        POINTNET_ROOT,
        "data_utils"
    )
)


# ============================================================
# ARGUMENTOS
# ============================================================

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
        '/home/manuel/Documents/GitHub/LiDAR/'
        'checkpoints/latest_model.pth'
    ),
    type=str
)


parser.add_argument(
    '--save_dir',
    default='./checkpoints',
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
    default=1e-4,
    type=float
)


parser.add_argument(
    '--num_points',
    default=4096,
    type=int
)


parser.add_argument(
    '--num_workers',
    default=0,
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


CLASS_NAMES = [
    "background",
    "person"
]


CLASS_WEIGHTS = [
    1.0,
    10.0
]


IGNORE_INDEX = -100


# ============================================================
# SEMILLA
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )


    np.random.seed(
        seed
    )


    torch.manual_seed(
        seed
    )


    torch.cuda.manual_seed(
        seed
    )


    torch.cuda.manual_seed_all(
        seed
    )


    torch.backends.cudnn.deterministic = True


    torch.backends.cudnn.benchmark = False


# ============================================================
# REPOSITORY RELU
# ============================================================

def inplace_relu(
    m
):

    classname = m.__class__.__name__


    if classname.find(
        'ReLU'
    ) != -1:

        m.inplace = True


# ============================================================
# CARGAR CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    path,
    optimizer=None,
    scheduler=None,
    device="cpu"
):

    if not os.path.exists(path):

        print(
            "\nNo se ha encontrado checkpoint:"
        )

        print(
            path
        )

        print(
            "Se comienza desde cero."
        )

        return 0, -1.0


    print(
        "\nCargando checkpoint:"
    )

    print(
        path
    )


    checkpoint = torch.load(

        path,

        map_location=device

    )


    # ========================================================
    # CARGAR PESOS DEL MODELO
    # ========================================================

    if "model_state_dict" in checkpoint:

        pretrained = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        pretrained = checkpoint[
            "state_dict"
        ]

    else:

        pretrained = checkpoint


    model_dict = model.state_dict()


    compatible = {}


    for key, value in pretrained.items():

        if key not in model_dict:

            continue


        if model_dict[key].shape != value.shape:

            continue


        compatible[key] = value


    model_dict.update(
        compatible
    )


    model.load_state_dict(
        model_dict
    )


    print(
        f"Pesos cargados: {len(compatible)}"
    )


    # ========================================================
    # CARGAR OPTIMIZER
    # ========================================================

    if (

        optimizer is not None

        and

        "optimizer_state_dict" in checkpoint

    ):

        optimizer.load_state_dict(

            checkpoint[
                "optimizer_state_dict"
            ]

        )


        print(
            "Estado del optimizer cargado"
        )


    # ========================================================
    # CARGAR SCHEDULER
    # ========================================================

    if (

        scheduler is not None

        and

        "scheduler_state_dict" in checkpoint

    ):

        scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]

        )


        print(
            "Estado del scheduler cargado"
        )


    # ========================================================
    # ÉPOCA
    # ========================================================

    start_epoch = checkpoint.get(

        "epoch",

        0

    )


    # ========================================================
    # MEJOR MIOU
    # ========================================================

    best_miou = checkpoint.get(

        "best_miou",

        checkpoint.get(

            "val_miou",

            -1.0

        )

    )


    print(

        f"Última época guardada: "
        f"{start_epoch}"

    )


    print(

        f"Mejor mIoU: "
        f"{best_miou:.4f}"

    )


    return start_epoch, best_miou

# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def update_confusion_matrix(
    confusion_matrix,
    predictions,
    labels,
    num_classes
):

    predictions = predictions.reshape(
        -1
    )


    labels = labels.reshape(
        -1
    )


    valid = (

        (labels >= 0)

        &

        (labels < num_classes)

    )


    predictions = predictions[
        valid
    ]


    labels = labels[
        valid
    ]


    indices = (

        num_classes * labels

        +

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


    confusion_matrix += (

        confusion.cpu()

    )


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(
    confusion_matrix
):

    confusion_matrix = (

        confusion_matrix.numpy()

    )


    true_positive = np.diag(

        confusion_matrix

    )


    false_positive = (

        confusion_matrix.sum(
            axis=0
        )

        -

        true_positive

    )


    false_negative = (

        confusion_matrix.sum(
            axis=1
        )

        -

        true_positive

    )


    total = confusion_matrix.sum()


    accuracy = (

        true_positive.sum()

        /

        max(
            total,
            1
        )

    )


    denominator_iou = (

        true_positive

        +

        false_positive

        +

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


    precision_denominator = (

        true_positive

        +

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


    recall_denominator = (

        true_positive

        +

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


    f1_denominator = (

        precision

        +

        recall

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


    metrics = {

        "accuracy": accuracy,

        "iou": iou,

        "miou": np.mean(iou),

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "confusion_matrix": confusion_matrix

    }


    return metrics


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


        points = points.transpose(

            2,

            1

        )


        pred, trans_feat = classifier(

            points

        )


        pred = pred.reshape(

            -1,

            NUM_CLASSES

        )


        labels = labels.reshape(

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

        confusion_matrix

    )


    mean_loss = (

        total_loss

        /

        max(

            len(val_loader),

            1

        )

    )


    return mean_loss, metrics


# ============================================================
# IMPRIMIR MÉTRICAS DE VALIDACIÓN
# ============================================================

def print_validation_metrics(
    metrics
):

    print(
        "\n"
    )


    print(
        "=============================="
    )


    print(
        "MÉTRICAS DE VALIDACIÓN"
    )


    print(
        "=============================="
    )


    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )


    print(
        f"mIoU: {metrics['miou']:.4f}"
    )


    print(
        f"Macro Precision: "
        f"{np.mean(metrics['precision']):.4f}"
    )


    print(
        f"Macro Recall: "
        f"{np.mean(metrics['recall']):.4f}"
    )


    print(
        f"Macro F1: "
        f"{np.mean(metrics['f1']):.4f}"
    )


    print(
        "\nPor clase:"
    )


    for class_name, iou, precision, recall, f1 in zip(

        CLASS_NAMES,

        metrics["iou"],

        metrics["precision"],

        metrics["recall"],

        metrics["f1"]

    ):

        print(

            f"  {class_name:<15} | "

            f"IoU: {iou:.4f} | "

            f"Precision: {precision:.4f} | "

            f"Recall: {recall:.4f} | "

            f"F1: {f1:.4f}"

        )


    print(
        "\nMatriz de confusión:"
    )


    print(
        metrics["confusion_matrix"]
    )


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


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(
        f"Dispositivo: {device}"
    )


# ========================================================
# MODELO
# ========================================================

    MODEL = importlib.import_module(
        'models.pointnet2_sem_seg'
    )


    classifier = MODEL.get_model(
        NUM_CLASSES
    )


    classifier.apply(
        inplace_relu
    )


    classifier.to(
        device
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

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

        f"Train samples: {len(train_dataset)}"

    )


    print(

        f"Val samples: {len(val_dataset)}"

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

        pin_memory=True

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        drop_last=False,

        pin_memory=True

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
# CARGAR CHECKPOINT COMPLETO
# ========================================================

    start_epoch, best_miou = load_checkpoint(

        classifier,

        args.checkpoint,

        optimizer,

        scheduler,

        device

    )


    # ========================================================
    # TRAINING
    # ========================================================

    #best_miou = -1.0


    epochs_without_improvement = 0


    for epoch in range(
        start_epoch,

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


        # ====================================================
        # TRAINING
        # ====================================================

        for points, labels in train_loader:


            points = points.to(

                device,

                non_blocking=True

            )


            labels = labels.to(

                device,

                non_blocking=True

            )


            points = points.transpose(

                2,

                1

            )


            optimizer.zero_grad(

                set_to_none=True

            )


            pred, trans_feat = classifier(

                points

            )


            pred = pred.reshape(

                -1,

                NUM_CLASSES

            )


            labels_flat = labels.reshape(

                -1

            )


            loss = criterion(

                pred,

                labels_flat

            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                classifier.parameters(),

                max_norm=1.0

            )


            optimizer.step()


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


        # ====================================================
        # MÉTRICAS DE TRAIN
        # ====================================================

        train_loss = (

            total_train_loss

            /

            max(

                len(train_loader),

                1

            )

        )


        train_metrics = calculate_metrics(

            train_confusion_matrix

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


        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(

            val_metrics["miou"]

        )


        current_lr = optimizer.param_groups[0]["lr"]


        # ====================================================
        # RESUMEN DE LA ÉPOCA
        # ====================================================

        print(

            "\n"

            + "=" * 70

        )


        print(

            f"Epoch {epoch + 1:03d}/{args.epoch}"

        )


        print(

            f"LR: {current_lr:.8f}"

        )


        print(

            f"Train Loss: {train_loss:.6f}"

        )


        print(

            f"Train Accuracy: "

            f"{train_metrics['accuracy']:.4f}"

        )


        print(

            f"Val Loss: {val_loss:.6f}"

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
        # MEJOR MODELO
        # ====================================================

        if (

            val_metrics["miou"]

            >

            best_miou

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

                f"Best mIoU: {best_miou:.4f}"

            )


        else:


            epochs_without_improvement += 1


        # ====================================================
        # LATEST CHECKPOINT
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

                "val_miou":

                    val_metrics["miou"],

                "val_loss":

                    val_loss

            },

            latest_path

        )


        # ====================================================
        # BLOQUE DE VALIDACIÓN
        # ====================================================

        print_validation_metrics(

            val_metrics

        )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (

            epochs_without_improvement

            >= args.patience

        ):


            print(

                "\nEarly stopping"

            )


            break


    # ========================================================
    # FINAL
    # ========================================================

    print(

        "\nEntrenamiento finalizado"

    )


    print(

        f"Mejor mIoU: {best_miou:.4f}"

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':

    main()