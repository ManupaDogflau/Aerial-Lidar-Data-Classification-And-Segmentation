import socket
import struct
import argparse
import numpy as np
import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "inference")
)

from inference_engine import InferenceEngine
from pointvector_model import PointVectorModel

HOST = "0.0.0.0"
PORT = 5005


def recv_exact(sock, size):
    data = b""

    while len(data) < size:
        packet = sock.recv(size - len(data))

        if not packet:
            raise ConnectionError("Conexión cerrada")

        data += packet

    return data


def parse_args():

    parser = argparse.ArgumentParser()

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

    ########################################################
    # Modelo
    ########################################################

    model = PointVectorModel(
        checkpoint=args.checkpoint,
        config=args.config,
        openpoints_root=args.openpoints,
        device=args.device
    )

    engine = InferenceEngine(model)

    ########################################################
    # Socket
    ########################################################

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"Esperando conexión en {HOST}:{PORT}...")

    conn, addr = server.accept()

    print(f"Conectado desde {addr}")

    ########################################################
    # Recepción
    ########################################################

    while True:

        try:

            header = recv_exact(conn, 4)

            n_points = struct.unpack("<I", header)[0]

            payload = recv_exact(conn, n_points * 3 * 4)

            cloud = np.frombuffer(
                payload,
                dtype=np.float32
            ).reshape(-1, 3)

            prediction, stats = engine.process(cloud)

            print("=" * 50)
            print(f"Frame      : {engine.frame_id - 1}")
            print(f"Puntos     : {len(cloud)}")

            for k, v in stats.items():
                print(f"{k:12}: {v}")

        except ConnectionError:
            print("Conexión cerrada.")
            break

        except Exception as e:
            print(f"[ERROR] {e}")

    conn.close()
    server.close()


if __name__ == "__main__":
    main()