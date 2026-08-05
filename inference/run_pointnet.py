import argparse


from benchmark import Benchmark

from inference_engine import InferenceEngine

from pointnet_model import PointNetModel

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
        "--pointnet",
        required=True
    )

    parser.add_argument(

        "--device",

        default="auto"

    )

    return parser.parse_args()


def main():

    args = parse_args()
    dataset = PCDDataset(args.dataset)

    model = PointNetModel(
        checkpoint=args.checkpoint,
        pointnet_root=args.pointnet,
        device=args.device
    )

    engine = InferenceEngine(model)

    benchmark = Benchmark(
        engine,
        dataset,
        output_csv="pointnet.csv"
    )

    results, summary = benchmark.run()

    benchmark.print_summary(summary)
    
if __name__ == "__main__":
    main()