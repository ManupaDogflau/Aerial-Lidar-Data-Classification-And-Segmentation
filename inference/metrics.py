import platform
import time
from datetime import datetime

import numpy as np
import psutil


class InferenceMetrics:

    def __init__(self):

        self.process = psutil.Process()

        self.has_cuda = False

        self.torch = None

        try:

            import torch

            self.torch = torch

            self.has_cuda = torch.cuda.is_available()

        except Exception:

            pass

    ##########################################################
    # Recoge métricas de una inferencia
    ##########################################################

    def collect(
        self,
        cloud,
        prediction,
        latency,
        frame=0,
        gt=None,
        confidence=None
    ):

        n_points = len(cloud)

        stats = {

            "frame": frame,

            "timestamp":
                datetime.now().isoformat(),

            "device":
                "cuda"
                if self.has_cuda
                else "cpu",

            "platform":
                platform.node(),

            "points":
                int(n_points),

            "latency_ms":
                latency * 1000,

            "fps":
                1.0 / latency,

            "ms_per_point":
                latency * 1000 / n_points,

            "points_per_second":
                n_points / latency,

            "cpu_percent":
                psutil.cpu_percent(),

            "ram_mb":
                self.process.memory_info().rss
                / (1024**2)
        }

        ##################################################

        if self.has_cuda:

            stats["gpu_memory_mb"] = (

                self.torch.cuda.max_memory_allocated()

                / (1024**2)

            )

        else:

            stats["gpu_memory_mb"] = None

        ##################################################

        freq = psutil.cpu_freq()

        if freq is not None:

            stats["cpu_freq_mhz"] = freq.current

        ##################################################

        try:

            temps = psutil.sensors_temperatures()

            if len(temps):

                first = list(temps.values())[0]

                stats["temperature"] = first[0].current

            else:

                stats["temperature"] = None

        except Exception:

            stats["temperature"] = None

        ##################################################

        if confidence is not None:

            stats["confidence"] = float(confidence)

        ##################################################

        if gt is not None:

            stats.update(

                self.segmentation_metrics(

                    prediction,

                    gt

                )

            )

        return stats

    ##########################################################
    # Métricas de segmentación
    ##########################################################

    def segmentation_metrics(

        self,

        pred,

        gt

    ):

        pred = np.asarray(pred).reshape(-1)

        gt = np.asarray(gt).reshape(-1)

        tp = np.sum(

            (pred == 1) & (gt == 1)

        )

        fp = np.sum(

            (pred == 1) & (gt == 0)

        )

        fn = np.sum(

            (pred == 0) & (gt == 1)

        )

        tn = np.sum(

            (pred == 0) & (gt == 0)

        )

        accuracy = (

            tp + tn

        ) / max(

            tp + tn + fp + fn,

            1

        )

        precision = tp / max(tp + fp, 1)

        recall = tp / max(tp + fn, 1)

        f1 = (

            2 * precision * recall

            /

            max(

                precision + recall,

                1e-8

            )

        )

        iou = tp / max(

            tp + fp + fn,

            1

        )

        return {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "iou": iou

        }

    ##########################################################
    # Resumen del benchmark
    ##########################################################

    def summary(self, results):

        if len(results) == 0:

            return {}

        latency = np.array(

            [

                r["latency_ms"]

                for r in results

            ]

        )

        fps = np.array(

            [

                r["fps"]

                for r in results

            ]

        )

        ram = np.array(

            [

                r["ram_mb"]

                for r in results

            ]

        )

        summary = {

            "frames":

                len(results),

            "latency_mean":

                latency.mean(),

            "latency_std":

                latency.std(),

            "latency_min":

                latency.min(),

            "latency_max":

                latency.max(),

            "latency_p95":

                np.percentile(latency,95),

            "latency_p99":

                np.percentile(latency,99),

            "fps_mean":

                fps.mean(),

            "fps_min":

                fps.min(),

            "fps_max":

                fps.max(),

            "ram_mean":

                ram.mean(),

            "ram_max":

                ram.max()

        }

        if "accuracy" in results[0]:

            for metric in [

                "accuracy",

                "precision",

                "recall",

                "f1",

                "iou"

            ]:

                summary[metric] = np.mean(

                    [

                        r[metric]

                        for r in results

                    ]

                )

        return summary