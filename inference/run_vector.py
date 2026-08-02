import argparse

from benchmark import Benchmark

from inference_engine import InferenceEngine

from pointvector_model import PointVectorModel

from pcd_dataset import PCDDataset


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--dataset",

        required=True,

        help="Carpeta con los .pcd"

    )

    parser.add_argument(

        "--checkpoint",

        required=True

    )

    parser.add_argument(

        "--config",

        required=True

    )

    parser.add_argument(

        "--openpoints",

        required=True

    )

    parser.add_argument(

        "--device",

        default="auto"

    )

    return parser.parse_args()


def main():

    args = parse_args()

    ######################################################

    dataset = PCDDataset(

        args.dataset

    )

    ######################################################

    model = PointVectorModel(

        checkpoint=args.checkpoint,

        config=args.config,

        openpoints_root=args.openpoints,

        device=args.device

    )

    ######################################################

    engine = InferenceEngine(

        model

    )

    ######################################################

    benchmark = Benchmark(

        engine,

        dataset

    )

    ######################################################

    results, summary = benchmark.run()

    ######################################################

    benchmark.print_summary(summary)


if __name__ == "__main__":

    main()