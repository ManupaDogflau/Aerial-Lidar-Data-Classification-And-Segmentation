from pathlib import Path
from datetime import datetime
import time

import numpy as np

from metrics import InferenceMetrics


class InferenceEngine:

    def __init__(
        self,
        model,
        output_dir=None,
        warmup=10
    ):

        self.model = model

        self.metrics = InferenceMetrics()

        self.frame_id = 0

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else None
        )

        self.warmup_iterations = warmup

        self._warmup_done = False
        
    def warmup(self, cloud):

        for _ in range(self.warmup_iterations):

            self.model.predict(cloud)

        self._warmup_done = True
        
    
    def process(
        self,
        cloud: np.ndarray,
        gt=None
    ):

        if not self._warmup_done:

            self.warmup(cloud)

        start = time.perf_counter()

        prediction = self.model.predict(cloud)

        latency = time.perf_counter() - start

        stats = self.metrics.collect(

            cloud=cloud,

            prediction=prediction,

            gt=gt,

            latency=latency,

            frame=self.frame_id
        )

        self.frame_id += 1

        return prediction, stats
    
    def run(self, cloud_generator):

        for cloud in cloud_generator:

            prediction, stats = self.process(cloud)

            yield prediction, stats
            
    def benchmark(
        self,
        cloud_generator,
        max_frames=None
    ):

        results = []

        for i, cloud in enumerate(cloud_generator):

            _, stats = self.process(cloud)

            results.append(stats)

            if (
                max_frames is not None
                and i + 1 >= max_frames
            ):
                break

        return self.metrics.summary(results)