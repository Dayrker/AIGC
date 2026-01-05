import os
import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        required=True,
        help="device list (like \"0, 1, 2, 3\").",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TI2V-5B",
        required=True,
        help="model name (like TI2V-5B/...).",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="480p",
        required=True,
        help="resolution ratio of vedio (like 480p/720p).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="NV",
        required=True,
        help="architecture (NV/DW).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="baseline",
        required=True,
        help="precision for inference (baseline/mxfp8/nvfp4).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        required=True,
        help="batch size (ie. 8).",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  # Set cuda devices first.
    return args