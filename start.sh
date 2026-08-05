#!/bin/bash
source /home/manuel/miniforge3/etc/profile.d/conda.sh
conda activate pointnet2
sudo $(which python) /home/manuel/lidar/LiDAR/main.py