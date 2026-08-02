import csv
from pathlib import Path

from tqdm import tqdm


class Benchmark:

    def __init__(
        self,
        engine,
        dataset,
        output_csv="benchmark.csv",
        verbose=True
    ):

        self.engine = engine
        self.dataset = dataset
        self.verbose = verbose

        self.output_csv = Path(output_csv)

    #########################################################

    def run(self):

        results = []

        iterator = self.dataset

        if self.verbose:

            iterator = tqdm(
                iterator,
                total=len(self.dataset),
                desc="Benchmark"
            )

        csv_file = None
        csv_writer = None

        try:

            csv_file = open(
                self.output_csv,
                "w",
                newline=""
            )

            for cloud in iterator:

                prediction, stats = self.engine.process(cloud)

                results.append(stats)

                # Crear el escritor usando las claves del primer resultado
                if csv_writer is None:

                    csv_writer = csv.DictWriter(
                        csv_file,
                        fieldnames=stats.keys()
                    )

                    csv_writer.writeheader()

                csv_writer.writerow(stats)
                csv_file.flush()

                if self.verbose:

                    tqdm.write(

                        f"Frame {stats['frame']:5d} | "

                        f"{stats['points']:6d} pts | "

                        f"{stats['latency_ms']:7.2f} ms | "

                        f"{stats['fps']:6.2f} FPS"

                    )

        finally:

            if csv_file is not None:

                csv_file.close()

        summary = self.engine.metrics.summary(results)

        return results, summary

    #########################################################

    @staticmethod
    def print_summary(summary):

        print()

        print("=" * 60)
        print("BENCHMARK")
        print("=" * 60)

        for key, value in summary.items():

            if isinstance(value, float):

                print(f"{key:25s}: {value:.3f}")

            else:

                print(f"{key:25s}: {value}")